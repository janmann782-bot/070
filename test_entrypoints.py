"""All hosting entrypoints must actually launch, never silently exit."""
import os
import subprocess
import sys
from pathlib import Path
import pytest

@pytest.mark.parametrize('filename',['run.py','main.py','bot.py'])
def test_launcher_reaches_configuration_error(filename):
    root=Path(__file__).resolve().parent
    env=os.environ.copy();env['BOT_TOKEN']='';env['PYTHONDONTWRITEBYTECODE']='1'
    result=subprocess.run([sys.executable,'-B',str(root/filename)],cwd=root,env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==1
    assert 'launcher started' in result.stdout
    assert 'BOT_TOKEN' in result.stderr
    assert 'Traceback' not in result.stderr
