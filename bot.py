# Bothost may autodetect bot.py. Route direct execution to the actual launcher.
if __name__ == '__main__':
    import sys
    sys.dont_write_bytecode = True
    from run import main as launch
    launch()
    raise SystemExit(0)

import asyncio
import io
import logging
import os
import secrets
import tempfile
from pathlib import Path
from collections import defaultdict
from datetime import date
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, BufferedInputFile, FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram import BaseMiddleware
from dotenv import load_dotenv
import config
from storage import Store, pack_masks
from state import World
from layers import Layers,read_png,png_bytes
from renderer import render
from orders import validate_markup,reference_hash
from exporter import export_video
from canon import advice
from progress import TelegramProgress
from cinematic import export_cinematic, preview
from lore_campaign import export_lore, preview_lore
from visual_settings import settings as visual_settings

HELP='''Аурелия • подневный фронт
/demo — новая война 28.06.2057, заменяет текущую
/newmap YYYY-MM-DD — затем стартовый PNG документом
/map — свежий PNG в исходном размере; рисуйте на нем
/map4k — просмотр шириной 3840, не для приказов
/turn DAYS WIDTH — одновременные приказы
/attack kefir DAYS WIDTH или /attack yogurt DAYS WIDTH
/wait DAYS — время без новых наступлений
/undo — отменить последнее событие
/doctor — переиграть журнал и восстановить кеш
/status — дата, контроль, оккупация, серая зона в пикселях
/terrain /cities /roads — затем PNG документом
/clearterrain /clearcities /clearroads — удалить слой
/layers — состояние слоев
/canon [kefir|yogurt] — советы по хронике
/export FPS WIDTH — все дни + начальный кадр
/export4k FPS — ширина 3840
/exportcurve FPS WIDTH — легкий изгиб готовых кадров
/exportcinema FPS WIDTH — анимации, UI Isaac, несколько кадров в день
/rerender FPS WIDTH — повторный cinematic рендер с текущими эффектами
/preview — кадр нового видеоинтерфейса
/lorepreview — кадр канонной войны в конце 2058
/lorewar [FPS WIDTH] — отдельный фильм 2057–2060 по лору
/effects — настройки эффектов кнопками
/effects seconds_per_day 0.4 — секунды на день (0.1..3)
/effects crf 18 — качество H.264 (14..26; меньше = качественнее)
/cancel — отменить ожидание загрузки

Кефирстан: #FF00FF, Йогуртстан: #00FF66
Линии 3–8 px: начните на своей территории, закончите в направлении противника. Можно несколько отдельных кривых. WIDTH — примерный оперативный масштаб в исходных пикселях. После /turn или /attack бот присылает свежую карту. Отправьте размеченный PNG как ФАЙЛ, без JPEG и изменения размера.
1 день = 1 кадр. Альтернативная история разрешена.'''

class OwnerGuard(BaseMiddleware):
    def __init__(self,owner): self.owner=owner
    async def __call__(self,handler,event,data):
        if self.owner and (event.from_user is None or event.from_user.id!=self.owner):
            await event.answer('Управление доступно владельцу бота')
            return
        return await handler(event,data)

class WarBot:
    def __init__(self,store,owner=0):
        self.store=store; self.owner=owner
        self.locks=defaultdict(asyncio.Lock)
        self.workers=asyncio.Semaphore(1)  # large native maps have bounded RAM
        self.dp=Dispatcher()
        self.dp.message.outer_middleware(OwnerGuard(owner))
        self.dp.callback_query.outer_middleware(OwnerGuard(owner))
        self.dp.message.register(self.command,F.text.startswith('/'))
        self.dp.message.register(self.document,F.document)
        self.dp.message.register(self.photo,F.photo)
        self.dp.callback_query.register(self.effect_button,F.data.startswith('visual:'))

    @staticmethod
    def effects_keyboard(value):
        names=dict(notifications='Названия и события',city_glow='Метки городов',attack_glow='Подсветка наступлений',postprocess='Цвет и резкость',vignette='Виньетка',curved='Изгиб карты')
        rows=[[InlineKeyboardButton(text=f"{'●' if value[key] else '○'} {label}",callback_data='visual:'+key)] for key,label in names.items()]
        return InlineKeyboardMarkup(inline_keyboard=rows)

    @staticmethod
    def effects_text(value):
        return (f"Аурелия • оформление v1.4.4\n{value['seconds_per_day']:g} с/день • CRF {value['crf']}\n"
                'Кнопки меняют оформление карты и видео.\n'
                '/effects seconds_per_day 0.4\n/effects crf 18\n'
                '/preview — проверить кадр\n/rerender 30 1920 — видео с новыми настройками')

    async def effect_button(self,query):
        if query.message is None:
            await query.answer();return
        chat=query.message.chat.id
        async with self.locks[chat]:
            key=query.data.partition(':')[2]
            if key not in ['notifications','city_glow','attack_glow','postprocess','vignette','curved']:
                await query.answer('Неизвестный эффект');return
            value=await self.work(visual_settings,self.store,chat)
            value=await self.work(visual_settings,self.store,chat,{key:not value[key]})
            await query.answer('Настройка сохранена')
            await query.message.edit_text(self.effects_text(value),reply_markup=self.effects_keyboard(value))

    async def work(self,fn,*args,**kwargs):
        async with self.workers: return await asyncio.to_thread(fn,*args,**kwargs)

    async def tracked_work(self,message,title,fn,*args,total=1,unit='дней',**kwargs):
        async with TelegramProgress(message,title,total,unit) as progress:
            return await self.work(fn,*args,progress=progress.report,**kwargs)

    async def send_map(self,message,width=None):
        def make(progress=None):
            progress(0,3,'Загрузка состояния')
            w,revision=self.store.load(message.chat.id)
            pending=self.store.pending(message.chat.id)
            stamp=f'r{revision}'
            if pending and pending[0]=='turn': stamp+=' / '+pending[1]['tag']
            progress(1,3,'Рендер политического слоя, рельефа и городов')
            grade=visual_settings(self.store,message.chat.id)['postprocess']
            if pending and pending[0]=='turn':
                grade=pending[1].get('map_postprocess',False)
            rgb=render(w,width,stamp=stamp,postprocess=grade)
            progress(2,3,'Кодирование PNG')
            data=png_bytes(rgb)
            progress(3,3,'Карта готова к отправке')
            return data,w.date.strftime('%d.%m.%Y')
        data,dt=await self.tracked_work(message,'Новая карта',make,total=3,unit='этапов')
        await message.answer_document(BufferedInputFile(data,filename='front_4k.png' if width else 'front.png'),caption=f'{dt} • '+('3840 px, только просмотр' if width else 'Исходный размер. Линии приказов рисуйте поверх этого PNG'))

    async def command(self,m:Message):
        async with self.locks[m.chat.id]:
            try: await self._command(m)
            except (ValueError,OSError) as e: await m.answer(str(e))
            except Exception:
                logging.exception('Command failed chat=%s',m.chat.id)
                await m.answer('Ошибка расчета. Незавершенная операция не записана; подробности в консоли')

    async def _command(self,m):
        parts=m.text.split(); cmd=parts[0].split('@')[0].lower(); args=parts[1:]; chat=m.chat.id
        if cmd in ['/start','/help']:
            await m.answer(HELP); return
        if cmd=='/effects':
            update=None
            if args==['reset']:
                from visual_settings import DEFAULTS
                update=dict(DEFAULTS)
            elif args:
                if len(args)!=2:raise ValueError('/effects PARAM VALUE или /effects reset')
                key,value=args
                if key in ['seconds_per_day','crf']:
                    value=float(value) if key=='seconds_per_day' else int(value)
                elif value in ['on','off']:
                    value=value=='on'
                else:raise ValueError('Значение: on/off; seconds_per_day: 0.1..3; crf: 14..26')
                update={key:value}
            value=await self.work(visual_settings,self.store,chat,update)
            await m.answer(self.effects_text(value),reply_markup=self.effects_keyboard(value));return
        if cmd=='/preview':
            if args:raise ValueError('/preview без параметров')
            rgb=await self.tracked_work(m,'Cinematic preview',preview,self.store,chat,total=3,unit='этапов')
            await m.answer_document(BufferedInputFile(png_bytes(rgb),filename='aurelia_cinematic_preview.png'),caption='UI v1.4.4 • карта для просмотра. Приказы рисуйте на /map');return
        if cmd=='/lorepreview':
            if args:raise ValueError('/lorepreview без параметров')
            rgb=await self.tracked_work(m,'Превью канонной войны',preview_lore,self.store,chat,total=3,unit='этапов')
            await m.answer_document(BufferedInputFile(png_bytes(rgb),filename='aurelia_lore_preview.png'),
                                    caption='Йогуртстанская война • 31.12.2058 • историческая визуализация');return
        if cmd=='/lorewar':
            if len(args) not in (0,2):raise ValueError('/lorewar [FPS WIDTH] (например /lorewar 24 1280)')
            fps,width=map(int,args) if args else (24,1280)
            fd,path=tempfile.mkstemp(suffix='.mp4');os.close(fd)
            try:
                result=await self.tracked_work(m,'Йогуртстанская война • канон',export_lore,self.store,chat,path,fps,width,unit='кадров')
                if Path(path).stat().st_size>config.MAX_EXPORT_BYTES:
                    raise ValueError('Фильм больше 49 MB; попробуйте /lorewar 20 960')
                await m.answer_document(FSInputFile(path,filename='aurelia_yogurtstan_war_2057_2060.mp4'),
                    caption=f"2057–2060 • {result['frames']} кадров • {width}×{result['height']} • {fps} FPS\\nЙогуртстан не переходит исходную границу Кефирстана")
            finally:Path(path).unlink(missing_ok=True)
            return
        if cmd=='/cancel':
            await self.work(self.store.cancel,chat); await m.answer('Ожидание загрузки отменено'); return
        if cmd=='/demo':
            if args: raise ValueError('/demo без параметров')
            def create():
                base=read_png((config.ROOT/'yogurtstan_demo_control.png').read_bytes())
                layers=Layers(**{k:read_png((config.ROOT/f'yogurtstan_{k}.png').read_bytes(),base.shape[:2]) for k in ['terrain','cities','roads']})
                self.store.create(chat,World.create(base,'2057-06-28',layers))
            await m.answer('Создаю новую Йогуртстанскую войну…')
            await self.work(create); await self.send_map(m); return
        if cmd=='/newmap':
            if len(args)!=1: raise ValueError('/newmap YYYY-MM-DD')
            date.fromisoformat(args[0])
            await self.work(self.store.set_pending,chat,'newmap',{'date':args[0]})
            await m.answer('Отправьте стартовую карту PNG как документ. Она заменит текущую войну. Пользовательские слои загрузите после нее'); return
        if cmd in ['/terrain','/cities','/roads']:
            await self.work(self.store.load,chat)
            if args: raise ValueError(cmd+' без параметров')
            await self.work(self.store.set_pending,chat,'layer',{'name':cmd[1:]})
            await m.answer('Отправьте PNG исходного размера как документ. Для городов нужен прозрачный фон. Дороги: прозрачный слой или белая маска на черном фоне'); return
        if cmd in ['/clearterrain','/clearcities','/clearroads']:
            name=cmd[len('/clear'):]
            await self.work(self.store.commit,chat,'layer',{'name':name})
            await m.answer('Слой удален; изменение сохранено в журнале'); return
        if cmd in ['/turn','/attack']:
            if cmd=='/attack':
                if len(args)!=3 or args[0] not in ['kefir','yogurt']: raise ValueError('/attack kefir|yogurt DAYS WIDTH')
                sides=[1 if args[0]=='kefir' else 2]; days,width=map(int,args[1:])
            else:
                if len(args)!=2: raise ValueError('/turn DAYS WIDTH')
                sides=[1,2]; days,width=map(int,args)
            if not 1<=days<=config.MAX_DAYS or not 2<=width<=config.MAX_WIDTH: raise ValueError('DAYS: 1..1460, WIDTH: 2..1200')
            def pending():
                w,rev=self.store.load(chat); tag=secrets.token_hex(4)
                grade=visual_settings(self.store,chat)['postprocess']
                reference=render(w,stamp=f'r{rev} / {tag}',postprocess=grade)
                self.store.set_pending(chat,'turn',{'days':days,'width':width,'sides':sides,'seed':secrets.randbits(31),'tag':tag,'map_postprocess':grade},reference,rev)
            await self.work(pending)
            await m.answer(f'Приказ на {days} дней, масштаб {width} px. Используйте СВЕЖИЙ PNG ниже')
            await self.send_map(m); return
        if cmd=='/wait':
            if len(args)!=1: raise ValueError('/wait DAYS')
            days=int(args[0])
            if not 1<=days<=config.MAX_DAYS: raise ValueError('DAYS: 1..1460')
            await self.tracked_work(m,'Расчет новых дней',self.store.commit,chat,'turn',{'days':days,'width':20,'seed':0},pack_masks({}),total=days)
            await self.send_map(m); return
        if cmd in ['/map','/map4k']:
            await self.send_map(m,3840 if cmd=='/map4k' else None); return
        if cmd=='/undo':
            await self.tracked_work(m,'Отмена и повторный расчет',self.store.undo,chat); await self.send_map(m); return
        if cmd=='/doctor':
            same,w=await self.tracked_work(m,'Проверка журнала',self.store.doctor,chat)
            await m.answer('Replay совпадает пиксель в пиксель' if same else 'Кеш расходился с журналом и восстановлен. Возьмите свежую /map'); return
        if cmd in ['/status','/layers','/canon']:
            w,_=await self.work(self.store.load,chat)
            if cmd=='/status':
                s=w.stats(); total=int(w.territory.sum())
                await m.answer(f"{w.date:%d.%m.%Y} • прошло {s['days']} дней\nКонтроль (без серой зоны):\nКефирстан: {s['kefir']:,} px ({s['kefir']/total:.2%})\nЙогуртстан: {s['yogurt']:,} px ({s['yogurt']/total:.2%})\nКефирская оккупация: {s['kefir_occupation']:,} px\nЙогуртская оккупация: {s['yogurt_occupation']:,} px\nContested: {s['contested']:,} px\nПлощадь измеряется в пикселях: масштаб км²/px не задан".replace(',',' '))
            elif cmd=='/layers':
                await m.answer(f'Исходное разрешение: {w.control.shape[1]}×{w.control.shape[0]}\n'+ '\n'.join(f'{k}: '+('загружен' if getattr(w.layers,k) is not None else 'нет') for k in ['terrain','cities','roads']))
            else:
                if len(args)>1 or (args and args[0] not in ['kefir','yogurt']): raise ValueError('/canon [kefir|yogurt]')
                await m.answer(advice(w.date,args[0] if args else None))
            return
        if cmd in ['/export','/export4k','/exportcurve','/exportcinema','/rerender']:
            if cmd=='/export4k':
                if len(args)!=1: raise ValueError('/export4k FPS')
                fps,width=int(args[0]),3840
            else:
                if len(args)!=2: raise ValueError(cmd+' FPS WIDTH')
                fps,width=map(int,args)
            cinematic=cmd in ['/exportcinema','/rerender']
            if not 1<=fps<=60 or not (320 if cinematic else 64)<=width<=4096: raise ValueError('FPS: 1..60, WIDTH: '+('320' if cinematic else '64')+'..4096')
            await self.work(self.store.load,chat)
            fd,path=tempfile.mkstemp(suffix='.mp4'); os.close(fd)
            try:
                if cinematic:
                    result=await self.tracked_work(m,'Cinematic: кадры и видео',export_cinematic,self.store,chat,path,fps,width,unit='кадров')
                else:
                    result=await self.tracked_work(m,'Рендер видео',export_video,self.store,chat,path,fps,width,cmd=='/exportcurve',unit='кадров')
                if Path(path).stat().st_size>config.MAX_EXPORT_BYTES:
                    raise ValueError('Видео превышает 49 MB. Экспортируйте с меньшей шириной')
                await m.answer_document(FSInputFile(path,filename='aurelia_cinematic.mp4' if cinematic else ('aurelia_curved.mp4' if cmd=='/exportcurve' else 'aurelia.mp4')),caption=f"{result['frames']} кадров • {result['width']}×{result['height']} • {fps} FPS"+(' • готовая временная линия' if result.get('cached') else ''))
            finally: Path(path).unlink(missing_ok=True)
            return
        raise ValueError('Неизвестная команда. /help')

    async def document(self,m:Message):
        async with self.locks[m.chat.id]:
            try:
                pending=await self.work(self.store.pending,m.chat.id)
                if not pending: raise ValueError('Сначала /turn, /attack, /newmap, /terrain, /cities или /roads')
                if m.document.file_size and m.document.file_size>config.MAX_UPLOAD_BYTES: raise ValueError('Максимум PNG: 19 MB')
                buf=io.BytesIO(); await m.bot.download(m.document,destination=buf); data=buf.getvalue()
                if len(data)>config.MAX_UPLOAD_BYTES: raise ValueError('Максимум PNG: 19 MB')
                kind,params,ref,digest,revision=pending
                if kind=='newmap':
                    def create(): self.store.create(m.chat.id,World.create(read_png(data),params['date']))
                    await self.work(create); await self.work(self.store.cancel,m.chat.id)
                elif kind=='layer':
                    await self.work(self.store.commit,m.chat.id,'layer',params,data)
                elif kind=='turn':
                    def validate():
                        w,rev=self.store.load(m.chat.id)
                        if rev!=revision: raise ValueError('Состояние изменилось. Начните новый /turn')
                        reference=read_png(ref)[:,:,:3]
                        if reference_hash(reference)!=digest: raise ValueError('Поврежден reference. Начните новый /turn')
                        masks=validate_markup(data,reference,params['sides'])
                        return pack_masks(masks)
                    payload=await self.work(validate)
                    await self.tracked_work(m,'Расчет новых кадров войны',self.store.commit,m.chat.id,'turn',{k:params[k] for k in ['days','width','seed']},payload,total=params['days'])
                await self.send_map(m)
            except (ValueError,OSError) as e: await m.answer(str(e))
            except Exception:
                logging.exception('Upload failed chat=%s',m.chat.id)
                await m.answer('Не удалось обработать файл. Операция не записана; подробности в консоли')
    async def photo(self,m):
        await m.answer('Отправьте PNG как документ / файл, без сжатия. Telegram-фото не подходит')

async def main():
    load_dotenv(config.ROOT/'.env')
    token=os.getenv('BOT_TOKEN','').strip()
    if not token or token=='...': raise ValueError('Заполните BOT_TOKEN в файле .env рядом с run.py')
    owner=int(os.getenv('OWNER_ID','0') or '0')
    filename=os.getenv('DATA_FILE','war_state.sqlite3')
    if Path(filename).name!=filename or filename in ['.','..']: raise ValueError('DATA_FILE должен быть именем файла, без папок')
    app=WarBot(Store(config.ROOT/filename),owner)
    async with Bot(token) as bot:
        await bot.delete_webhook(drop_pending_updates=False)
        await app.dp.start_polling(bot,close_bot_session=False)
