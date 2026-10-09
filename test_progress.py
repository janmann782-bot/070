import asyncio
import time
from aiogram import Bot
from aiogram.methods import EditMessageText,SendDocument,SendMessage
from aiogram.exceptions import TelegramBadRequest
from progress import TelegramProgress
from bot import WarBot
from test_bot import FakeTelegram,feed
from test_storage import store,attack,assert_same


def test_progress_during_thread_work_is_throttled_and_finishes():
    class Message:
        def __init__(self):self.edits=[];self.answers=[]
        async def answer(self,text):self.answers.append(text);return self
        async def edit_text(self,text):self.edits.append(text)
    async def scenario():
        message=Message()
        async with TelegramProgress(message,'Видео',100,'кадров',interval=.025) as progress:
            def worker():
                for frame in range(1,101):
                    progress.report(frame,100,'Рендер кадров');time.sleep(.002)
            await asyncio.to_thread(worker)
        assert len(message.answers)==1
        assert 1<len(message.edits)<20  # real in-flight edits, coalesced worker notifications
        assert any('100%' not in text for text in message.edits[:-1])
        assert '100%' in message.edits[-1] and '100/100 кадров' in message.edits[-1]
        assert progress.task.done()
    asyncio.run(scenario())


def test_progress_error_never_reports_success_or_leaves_pump_running():
    class Message:
        def __init__(self):self.edits=[]
        async def answer(self,text):return self
        async def edit_text(self,text):self.edits.append(text)
    async def scenario():
        message=Message();progress=TelegramProgress(message,'Расчет',10)
        try:
            async with progress:
                progress.report(3,10,'День рассчитан')
                await asyncio.sleep(0)
                raise ValueError('failed')
        except ValueError:pass
        assert 'не завершен' in message.edits[-1] and '100%' not in message.edits[-1]
        assert progress.task.done()
    asyncio.run(scenario())


def test_telegram_status_failure_does_not_fail_the_operation():
    class Message:
        async def answer(self,text):return self
        async def edit_text(self,text):
            raise TelegramBadRequest(method=EditMessageText(chat_id=1,message_id=1,text=text),message='message to edit not found')
    async def scenario():
        async with TelegramProgress(Message(),'Видео') as progress:
            progress.report(1,1,'Кадр готов');await asyncio.sleep(0)
        assert progress.task.done()
    asyncio.run(scenario())


def test_wait_export_map_and_replay_progress_through_dispatcher(store):
    async def scenario():
        transport=FakeTelegram();bot=Bot('123456:abcdefghijklmnopqrstuvwxyzABCDEFGHI',session=transport)
        app=WarBot(store)
        await feed(app,bot,'/wait 2')
        await feed(app,bot,'/doctor')
        await feed(app,bot,'/export 10 320')
        await feed(app,bot,'/undo')
        edits=[m.text for m in transport.sent if isinstance(m,EditMessageText)]
        assert any('2/2 дней' in text and '100%' in text for text in edits)
        assert any('3/3 кадров' in text and '100%' in text for text in edits)
        assert any('3/3 этапов' in text and '100%' in text for text in edits)
        assert any('Проверка журнала' in text and '100%' in text for text in edits)
        assert any('Отмена и повторный расчет' in text and '100%' in text for text in edits)
        assert any(isinstance(m,SendDocument) and '3 кадров' in (m.caption or '') for m in transport.sent)
        assert store.load(1)[0].elapsed==0
        await bot.session.close()
    asyncio.run(scenario())


def test_simulation_and_replay_progress_do_not_change_state(store):
    expected=attack(store);updates=[]
    actual=store.replay(1,progress=lambda *args:updates.append(args))
    assert_same(expected,actual)
    assert [done for done,total,stage in updates if stage.startswith('Переигран день')]==list(range(1,7))
    assert all(total==6 for done,total,stage in updates)
    waited=[]
    store.commit(1,'turn',{'days':2,'width':20,'seed':0},progress=lambda *args:waited.append(args))
    assert [done for done,total,stage in waited if stage.startswith('Рассчитан день')]==[1,2]
    assert_same(store.load(1)[0],store.replay(1))
