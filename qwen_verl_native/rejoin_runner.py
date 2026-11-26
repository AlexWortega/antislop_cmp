"""Replace a dead launcher on a live rental: keep the tunnel up and run native_runner, without re-uploading."""
import os,subprocess,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'qwen_scaffold_longrun'
sys.path.insert(0,str(OLD));import run_loop as t
t.ROOT=ROOT;PORT=os.environ['ANTISLOP_COORDINATOR_PORT'];STOP_TUNNEL=threading.Event()
def tunnel_loop():
 args=t.ssh_args()
 with (ROOT/'tunnel.log').open('a') as log:
  while not STOP_TUNNEL.is_set():
   child=subprocess.Popen(args[:-1]+['-N','-o','ExitOnForwardFailure=yes','-L','0.0.0.0:18974:127.0.0.1:9001','-R','127.0.0.1:18975:127.0.0.1:'+PORT,args[-1]],stdout=log,stderr=subprocess.STDOUT)
   while child.poll() is None and not STOP_TUNNEL.is_set():STOP_TUNNEL.wait(5)
   if child.poll() is None:child.terminate()
   else:STOP_TUNNEL.wait(5)
threading.Thread(target=tunnel_loop,daemon=True).start()
code=1
try:code=subprocess.run([sys.executable,'-u',str(ROOT/'native_runner.py')],env=os.environ).returncode
finally:STOP_TUNNEL.set();(ROOT/'STOP').touch()
sys.exit(code)
