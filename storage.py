"""One SQLite file: atomic event log, cache and pending orders. No session folders."""
import io
import json
import sqlite3
import secrets
from contextlib import contextmanager
import numpy as np
from state import World
from layers import Layers, read_png
from engine import simulate, update_contested
from orders import reference_hash
import config

class MissingSession(ValueError): pass


def pack_world(w):
    buf=io.BytesIO(); arrays={k:getattr(w,k) for k in ['background','homeland','initial_control','control','battle_age','encirclement_age','disputed']}
    for k in ['terrain','cities','roads']:
        layer=getattr(w.layers,k)
        if layer is not None: arrays[k]=layer
    arrays['meta']=np.array(json.dumps({'start_date':w.start_date,'elapsed':w.elapsed,'engine':config.ENGINE_VERSION,'physics':w.layers.settings}))
    np.savez_compressed(buf,**arrays)
    return buf.getvalue()


def unpack_world(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as a:
        meta=json.loads(str(a['meta']))
        if meta['engine']!=config.ENGINE_VERSION: raise ValueError('Несовместимая версия движка')
        layer=Layers(settings=meta['physics'],**{k:a[k].copy() if k in a else None for k in ['terrain','cities','roads']})
        layer.rebuild(a['control'].shape)
        args={k:a[k].copy() for k in ['background','homeland','initial_control','control','battle_age','encirclement_age','disputed']}
    args['homeland'].flags.writeable=False; args['initial_control'].flags.writeable=False
    return World(**args,layers=layer,start_date=meta['start_date'],elapsed=meta['elapsed'])


def pack_masks(masks):
    b=io.BytesIO(); np.savez_compressed(b,**{str(s):m for s,m in masks.items()}); return b.getvalue()


def unpack_masks(blob):
    if blob is None: return {}
    with np.load(io.BytesIO(blob),allow_pickle=False) as a: return {int(k):a[k].copy() for k in a.files}


class Store:
    def __init__(self,path):
        self.path=str(path)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS sessions(chat_id INTEGER PRIMARY KEY, initial BLOB NOT NULL,
              cache BLOB NOT NULL, revision INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS operations(id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL,
              kind TEXT NOT NULL, params TEXT NOT NULL, payload BLOB,
              FOREIGN KEY(chat_id) REFERENCES sessions(chat_id) ON DELETE CASCADE);
            CREATE INDEX IF NOT EXISTS operations_chat ON operations(chat_id,id);
            CREATE TABLE IF NOT EXISTS pending(chat_id INTEGER PRIMARY KEY, kind TEXT NOT NULL,
              params TEXT NOT NULL, reference BLOB, hash TEXT, revision INTEGER);
            ''')
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=60)
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db: yield db
        finally:
            db.close()

    def create(self,chat_id,world):
        blob=pack_world(world)
        with self.connect() as db:
            db.execute('DELETE FROM pending WHERE chat_id=?',(chat_id,))
            db.execute('DELETE FROM sessions WHERE chat_id=?',(chat_id,))
            db.execute('INSERT INTO sessions(chat_id,initial,cache) VALUES(?,?,?)',(chat_id,blob,blob))

    def load(self,chat_id):
        with self.connect() as db: row=db.execute('SELECT cache,revision FROM sessions WHERE chat_id=?',(chat_id,)).fetchone()
        if not row: raise MissingSession('Сначала /demo или /newmap YYYY-MM-DD')
        return unpack_world(row[0]),row[1]

    def events(self,chat_id):
        with self.connect() as db: rows=db.execute('SELECT kind,params,payload FROM operations WHERE chat_id=? ORDER BY id',(chat_id,)).fetchall()
        return [(kind,json.loads(params),payload) for kind,params,payload in rows]

    @staticmethod
    def apply(world,kind,params,payload,on_day=None):
        if kind=='turn':
            simulate(world,unpack_masks(payload),params['days'],params['width'],params['seed'],on_day)
        elif kind=='layer':
            name=params['name']; data=None if payload is None else read_png(payload,world.control.shape)
            setattr(world.layers,name,data); world.layers.rebuild(world.control.shape)
        else: raise ValueError('Неизвестный тип события')

    def commit(self,chat_id,kind,params,payload=None):
        world,revision=self.load(chat_id)
        self.apply(world,kind,params,payload)
        blob=pack_world(world)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            current=db.execute('SELECT revision FROM sessions WHERE chat_id=?',(chat_id,)).fetchone()
            if not current or current[0]!=revision: raise ValueError('Сессия изменилась; повторите команду')
            db.execute('INSERT INTO operations(chat_id,kind,params,payload) VALUES(?,?,?,?)',(chat_id,kind,json.dumps(params),payload))
            db.execute('UPDATE sessions SET cache=?,revision=revision+1 WHERE chat_id=?',(blob,chat_id))
            db.execute('DELETE FROM pending WHERE chat_id=?',(chat_id,))
        return world

    def replay(self,chat_id,on_start=None,on_day=None,omit_last=False):
        with self.connect() as db: row=db.execute('SELECT initial FROM sessions WHERE chat_id=?',(chat_id,)).fetchone()
        if not row: raise MissingSession('Нет войны')
        w=unpack_world(row[0])
        if on_start: on_start(w)
        events=self.events(chat_id)
        for kind,params,payload in (events[:-1] if omit_last else events): self.apply(w,kind,params,payload,on_day)
        return w

    def undo(self,chat_id):
        _,revision=self.load(chat_id)
        if not self.events(chat_id): raise ValueError('Журнал пуст')
        w=self.replay(chat_id,omit_last=True); blob=pack_world(w)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT revision FROM sessions WHERE chat_id=?',(chat_id,)).fetchone()[0]!=revision: raise ValueError('Сессия изменилась')
            db.execute('DELETE FROM operations WHERE id=(SELECT MAX(id) FROM operations WHERE chat_id=?)',(chat_id,))
            db.execute('UPDATE sessions SET cache=?,revision=revision+1 WHERE chat_id=?',(blob,chat_id))
            db.execute('DELETE FROM pending WHERE chat_id=?',(chat_id,))
        return w

    def doctor(self,chat_id):
        # Does not trust cache: can repair even a corrupt compressed BLOB.
        with self.connect() as db: row=db.execute('SELECT cache,revision FROM sessions WHERE chat_id=?',(chat_id,)).fetchone()
        if not row: raise MissingSession('Нет войны')
        w=self.replay(chat_id)
        try:
            cached=unpack_world(row[0])
            same=cached.elapsed==w.elapsed and cached.start_date==w.start_date and all(np.array_equal(getattr(cached,k),getattr(w,k)) for k in ['background','initial_control','control','homeland','battle_age','encirclement_age','disputed'])
            same &= cached.layers.settings==w.layers.settings
            same &= all((getattr(cached.layers,k) is None and getattr(w.layers,k) is None) or (getattr(cached.layers,k) is not None and getattr(w.layers,k) is not None and np.array_equal(getattr(cached.layers,k),getattr(w.layers,k))) for k in ['terrain','cities','roads'])
        except Exception: same=False
        blob=pack_world(w)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT revision FROM sessions WHERE chat_id=?',(chat_id,)).fetchone()[0]!=row[1]: raise ValueError('Сессия изменилась')
            db.execute('UPDATE sessions SET cache=?,revision=revision+? WHERE chat_id=?',(blob,int(not same),chat_id))
            if not same: db.execute('DELETE FROM pending WHERE chat_id=?',(chat_id,))
        return bool(same),w

    def set_pending(self,chat_id,kind,params,reference=None,revision=None):
        from layers import png_bytes
        blob=png_bytes(reference) if reference is not None else None
        digest=reference_hash(reference) if reference is not None else None
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO pending VALUES(?,?,?,?,?,?)',(chat_id,kind,json.dumps(params),blob,digest,revision))
    def pending(self,chat_id):
        with self.connect() as db: row=db.execute('SELECT kind,params,reference,hash,revision FROM pending WHERE chat_id=?',(chat_id,)).fetchone()
        return None if not row else (row[0],json.loads(row[1]),row[2],row[3],row[4])
    def cancel(self,chat_id):
        with self.connect() as db: db.execute('DELETE FROM pending WHERE chat_id=?',(chat_id,))
