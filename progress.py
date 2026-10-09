"""One edited Telegram message; worker threads only publish small counters."""
import asyncio
import logging
from contextlib import suppress
from aiogram.exceptions import TelegramRetryAfter,TelegramAPIError


class TelegramProgress:
    def __init__(self,message,title,total=1,unit='этапов',interval=2.0):
        self.message=message;self.title=title;self.total=total;self.unit=unit
        self.interval=interval;self.done=0;self.stage='Ожидание расчета'
        self.closed=False;self.task=None;self.last_text=None;self.retry_at=0

    def text(self,finished=False,failed=False):
        percent=100 if finished else min(99,int(100*self.done/max(1,self.total)))
        filled=12 if finished else percent*12//100
        bar='█'*filled+'░'*(12-filled)
        stage='Готово' if finished else ('Расчет не завершен' if failed else self.stage)
        return f'{self.title}\n[{bar}] {percent}%\n{self.done}/{self.total} {self.unit}\n{stage}'

    async def __aenter__(self):
        self.loop=asyncio.get_running_loop()
        self.last_text=self.text()
        self.status=await self.message.answer(self.last_text)
        self.task=asyncio.create_task(self.pump())
        return self

    def report(self,done,total,stage):
        # Called by NumPy/FFmpeg workers: never use Telegram or retain a World here.
        if self.closed:return
        def update():
            if not self.closed:
                self.total=max(0,int(total));self.done=max(0,min(int(done),self.total));self.stage=stage
        self.loop.call_soon_threadsafe(update)

    async def edit(self,text,final=False):
        if text==self.last_text:return
        if not final and self.loop.time()<self.retry_at:return
        try:
            await self.status.edit_text(text)
            self.last_text=text
        except TelegramRetryAfter as e:
            self.retry_at=self.loop.time()+e.retry_after
            # Honor Telegram's flood-control delay for the single final update.
            if final:
                if e.retry_after>60:
                    logging.warning('Final progress update postponed by Telegram flood control')
                    return
                await asyncio.sleep(e.retry_after)
                try:
                    await self.status.edit_text(text);self.last_text=text
                except TelegramAPIError:logging.warning('Final progress update unavailable')
        except TelegramAPIError:
            # A removed status message or network/API failure cannot spoil a calculation.
            logging.warning('Progress update unavailable',exc_info=True)

    async def pump(self):
        while True:
            await asyncio.sleep(self.interval)
            await self.edit(self.text())

    async def __aexit__(self,exc_type,exc,tb):
        self.closed=True
        self.task.cancel()
        with suppress(asyncio.CancelledError):await self.task
        if exc_type is None:self.done=self.total
        await self.edit(self.text(finished=exc_type is None,failed=exc_type is not None),final=True)
        return False
