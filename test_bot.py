"""Aiogram dispatcher exercised offline through a mock Telegram transport."""
import asyncio
import io
from datetime import datetime,timezone
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.types import Update,Message,Chat,User,File
from aiogram.methods import SendMessage,SendDocument,GetFile
from bot import WarBot
from test_storage import store
from layers import png_bytes
from renderer import render
from test_renderer import markup

class FakeTelegram(BaseSession):
    def __init__(self): super().__init__(); self.sent=[]; self.upload=b''
    async def close(self): pass
    async def make_request(self,bot,method,timeout=None):
        if isinstance(method,GetFile): return File(file_id='file',file_unique_id='unique',file_path='mock.png')
        self.sent.append(method)
        if isinstance(method,(SendMessage,SendDocument)):
            return Message(message_id=len(self.sent),date=datetime.now(timezone.utc),chat=Chat(id=method.chat_id,type='private'),text=getattr(method,'text',None))
        return True
    async def stream_content(self,url,headers=None,timeout=30,chunk_size=65536,raise_for_status=True):
        yield self.upload

async def feed(app,bot,text=None,document=False,uid=7):
    params=dict(message_id=1,date=datetime.now(timezone.utc),chat=Chat(id=1,type='private'),from_user=User(id=uid,is_bot=False,first_name='Test'))
    if text: params['text']=text
    if document: params['document']=dict(file_id='file',file_unique_id='unique',file_name='front.png',mime_type='image/png',file_size=len(bot.session.upload))
    await app.dp.feed_update(bot,Update(update_id=1,message=Message(**params)))

def test_owner_and_dispatcher_turn_upload(store):
    async def scenario():
        transport=FakeTelegram(); bot=Bot('123456:abcdefghijklmnopqrstuvwxyzABCDEFGHI',session=transport)
        app=WarBot(store,owner=7)
        await feed(app,bot,'/wait 2',uid=8)
        assert store.load(1)[0].elapsed==0
        assert 'владельцу' in transport.sent[-1].text
        await feed(app,bot,'/turn 3 45')
        pending=store.pending(1); assert pending[0]=='turn'
        from layers import read_png
        ref=read_png(pending[2])[:,:,:3]
        transport.upload=markup(ref)
        await feed(app,bot,document=True)
        assert store.load(1)[0].elapsed==3 and store.pending(1) is None
        await feed(app,bot,'/doctor')
        assert 'пиксель' in transport.sent[-1].text
        await feed(app,bot,'/status')
        assert '3 дней' in transport.sent[-1].text
        await bot.session.close()
    asyncio.run(scenario())

def test_invalid_upload_does_not_consume_pending(store):
    async def scenario():
        transport=FakeTelegram(); bot=Bot('123456:abcdefghijklmnopqrstuvwxyzABCDEFGHI',session=transport)
        app=WarBot(store)
        await feed(app,bot,'/attack kefir 2 45')
        transport.upload=b'JPEG'
        await feed(app,bot,document=True)
        assert 'PNG' in transport.sent[-1].text
        assert store.pending(1) is not None and store.events(1)==[]
        await feed(app,bot,'/cancel'); assert store.pending(1) is None
        await bot.session.close()
    asyncio.run(scenario())
