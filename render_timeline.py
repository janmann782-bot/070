"""Read-only replay once; subsequent exports use compressed sparse daily deltas.

Presentation cache is disposable. The operation log remains authoritative.
No full-resolution daily PNGs or per-war folders are stored.
"""
import hashlib
import io
import json
import numpy as np
from storage import unpack_world, MissingSession, Store
from settlements import Settlements
from city_catalog import present_info

VERSION = 'cinematic-1.4.1'


def pack(arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def unpack(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


class Timeline:
    def __init__(self, store, chat_id):
        self.store, self.chat_id = store, chat_id
        with store.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS render_cache(chat_id INTEGER PRIMARY KEY, digest TEXT NOT NULL,
              metadata TEXT NOT NULL, FOREIGN KEY(chat_id) REFERENCES sessions(chat_id) ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS render_frames(chat_id INTEGER NOT NULL, day INTEGER NOT NULL,
              delta BLOB NOT NULL, info TEXT NOT NULL, checksum TEXT NOT NULL, PRIMARY KEY(chat_id,day),
              FOREIGN KEY(chat_id) REFERENCES render_cache(chat_id) ON DELETE CASCADE);
            ''')
        with store.connect() as db:
            row = db.execute('SELECT initial FROM sessions WHERE chat_id=?',(chat_id,)).fetchone()
        if row is None:
            raise MissingSession('Нет войны')
        self.initial = row[0]
        self.events = store.events(chat_id)
        h = hashlib.sha256(VERSION.encode()+self.initial)
        for kind, params, payload in self.events:
            h.update(kind.encode()); h.update(json.dumps(params, sort_keys=True).encode())
            h.update(payload or b'')
        self.digest = h.hexdigest()
        self.days = sum(p['days'] for k,p,_ in self.events if k=='turn')

    def ensure(self, progress=None):
        with self.store.connect() as db:
            row = db.execute('SELECT digest,metadata FROM render_cache WHERE chat_id=?',(self.chat_id,)).fetchone()
            cached = db.execute('SELECT delta,info,checksum FROM render_frames WHERE chat_id=? ORDER BY day',(self.chat_id,)).fetchall()
        if row and row[0] == self.digest and len(cached) == self.days+1:
            try:
                for delta, info, checksum in cached:
                    if hashlib.sha256(delta+info.encode()).hexdigest() != checksum:
                        raise ValueError('cache checksum')
                    unpack(delta)
                if progress: progress(1,1,'Готовая временная линия: фронт не пересчитывается')
                return False
            except (ValueError, OSError, EOFError):
                pass
        self._build(progress)
        return True

    def _build(self, progress):
        world = unpack_world(self.initial)
        previous = world.control.copy(); old_grey = world.disputed.copy()
        cities = Settlements(world); owners = cities.owners(world)
        # Retain last confirmed owner while a marker is contested.
        activity = []; rows = []; layer_ops = []; physics = world.layers.settings
        def record(w):
            nonlocal previous, old_grey, owners
            changed = np.flatnonzero(w.control != previous).astype(np.uint32)
            grey = np.flatnonzero(w.disputed != old_grey).astype(np.uint32)
            fresh = np.flatnonzero(w.fresh_capture).astype(np.uint32) if w.fresh_capture is not None else np.empty(0,np.uint32)
            delta = pack(dict(changed=changed, values=w.control.ravel()[changed], grey=grey, fresh=fresh))
            current, counts, captures = cities.describe(w, owners)
            owners.update({key:value for key,value in current.items() if value})
            if w.elapsed: activity.append(int(len(fresh)))
            info = dict(label=w.date.strftime('%d.%m.%Y'), days=w.elapsed, stats=w.stats(),
                        territory=int(w.territory.sum()), cities=counts, captures=captures,
                        activity=activity[-60:], layer_ops=list(layer_ops), physics=dict(physics),
                        source_size=[w.control.shape[1],w.control.shape[0]])
            text = json.dumps(info)
            rows.append((self.chat_id,w.elapsed,delta,text,hashlib.sha256(delta+text.encode()).hexdigest()))
            previous = w.control.copy(); old_grey = w.disputed.copy()
            if progress: progress(w.elapsed,self.days,f'Подготовка визуальной истории: {w.date:%d.%m.%Y}')
        record(world)
        for index, (kind,params,payload) in enumerate(self.events):
            if kind=='layer':
                Store.apply(world,kind,params,payload)
                layer_ops.append(index)
                if params['name']=='cities':
                    cities = Settlements(world); owners = cities.owners(world)
            else:
                if 'physics' in params: physics = params['physics']
                Store.apply(world,kind,params,payload,on_day=record)
        # A layer loaded after the last turn must also appear on the final frame.
        if rows:
            data = json.loads(rows[-1][3]); data['layer_ops'] = list(layer_ops)
            data['cities'] = cities.describe(world, owners)[1]
            text = json.dumps(data); last = rows[-1]
            rows[-1] = (*last[:3],text,hashlib.sha256(last[2]+text.encode()).hexdigest())
        # Verify the log hasn't changed before publishing a complete cache.
        if Timeline(self.store,self.chat_id).digest != self.digest:
            raise ValueError('Журнал изменился во время подготовки видео')
        with self.store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM render_cache WHERE chat_id=?',(self.chat_id,))
            db.execute('INSERT INTO render_cache VALUES(?,?,?)',(self.chat_id,self.digest,json.dumps({'days':self.days,'version':VERSION})))
            db.executemany('INSERT INTO render_frames VALUES(?,?,?,?,?)',rows)

    def frames(self):
        world = unpack_world(self.initial)
        applied = set()
        recent = []
        with self.store.connect() as db:
            cursor = db.execute('SELECT delta,info FROM render_frames WHERE chat_id=? ORDER BY day',(self.chat_id,))
            for blob, text in cursor:
                info = json.loads(text); data = unpack(blob)
                for index in info['layer_ops']:
                    if index not in applied:
                        Store.apply(world,*self.events[index]); applied.add(index)
                world.layers.settings = info['physics']
                world.control.ravel()[data['changed']] = data['values']
                world.disputed.ravel()[data['grey']] ^= True
                world.fresh_capture = np.zeros(world.control.shape,bool)
                world.fresh_capture.ravel()[data['fresh']] = True
                world.elapsed = info['days']
                info = present_info(info, recent)
                recent = info['recent']
                yield world, info
