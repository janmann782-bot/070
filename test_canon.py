from datetime import date,timedelta
from canon import advice,PHASES,START,END

def test_canon_dates_and_starovir():
    assert START==date(2057,6,28) and END==date(2060,11,12)
    for d in [date(2057,7,11),date(2057,7,18)]:
        text=advice(d,'kefir')
        assert 'Старовир' in text and 'через Новомир' in text
    for d,_,_,_,_ in PHASES:
        assert 'Фаза:' in advice(d)
    assert 'После официального' in advice(END+timedelta(days=1))
    assert 'До начала' in advice(START-timedelta(days=1))
