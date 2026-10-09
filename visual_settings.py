"""Per-chat presentation preferences, separate from saved physics settings."""
import json

DEFAULTS = dict(notifications=True, city_glow=True, attack_glow=True,
               postprocess=True, vignette=True, curved=False, seconds_per_day=.4, crf=18)


def validate(values):
    result = dict(DEFAULTS)
    for key, value in values.items():
        if key not in DEFAULTS:
            raise ValueError('Неизвестный визуальный параметр: '+key)
        if isinstance(DEFAULTS[key], bool):
            if not isinstance(value, bool):
                raise ValueError(key+': on или off')
        elif key == 'seconds_per_day':
            if not .1 <= float(value) <= 3:
                raise ValueError('seconds_per_day: 0.1..3')
            value = float(value)
        elif key == 'crf':
            if isinstance(value, bool) or not isinstance(value, int) or not 14 <= value <= 26:
                raise ValueError('crf: целое число 14..26')
        result[key] = value
    return result


def settings(store, chat_id, update=None):
    with store.connect() as db:
        db.execute('CREATE TABLE IF NOT EXISTS visual_settings(chat_id INTEGER PRIMARY KEY, value TEXT NOT NULL)')
        row = db.execute('SELECT value FROM visual_settings WHERE chat_id=?', (chat_id,)).fetchone()
        value = validate(json.loads(row[0]) if row else {})
        if update is not None:
            value = validate(dict(value, **update))
            db.execute('INSERT OR REPLACE INTO visual_settings VALUES(?,?)', (chat_id,json.dumps(value)))
    return value
