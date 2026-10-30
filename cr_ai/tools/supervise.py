"""训练守护：训练进程异常退出（例如显卡驱动报错 nvlddmkm 153 后 CUDA 上下文损坏）时，等显卡恢复再自动续训。

    python -m cr_ai.tools.supervise [--max-restarts 20] [--wait 60] -- python -m cr_ai.train.ppo --run runs/ppo1 --resume

- 子命令正常结束（返回码 0）就退出；
- 异常退出：记一行到 --log（默认 logs/supervise.log），等 --wait 秒再重启（子命令里要带 --resume）；
- 连续 --fast-fails 次在 --min-uptime 秒内就挂掉，说明不是偶发问题，停下来等人看。
"""
import argparse
import subprocess
import sys
import time

from ..config import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-restarts', type=int, default=20)
    ap.add_argument('--wait', type=float, default=60.0)
    ap.add_argument('--min-uptime', type=float, default=180.0)
    ap.add_argument('--fast-fails', type=int, default=4)
    ap.add_argument('--log', default='logs/supervise.log')
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == '--' else a.cmd
    if not cmd:
        ap.error('缺少要守护的命令')
    log = open(a.log if a.log.startswith(('/', '\\')) or ':' in a.log else '%s/%s' % (ROOT, a.log), 'a', encoding='utf-8')

    def note(msg):
        line = '%s %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg)
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()

    restarts, fast = 0, 0
    while True:
        t0 = time.time()
        note('启动：%s' % ' '.join(cmd))
        rc = subprocess.call(cmd, cwd=ROOT)
        up = time.time() - t0
        if rc == 0:
            note('正常结束')
            return 0
        restarts += 1
        fast = fast + 1 if up < a.min_uptime else 0
        note('异常退出（返回码 %d，运行 %.0f 秒），第 %d 次重启' % (rc, up, restarts))
        if restarts > a.max_restarts or fast >= a.fast_fails:
            note('重启次数过多或连续快速失败，停止守护')
            return rc
        time.sleep(a.wait)


if __name__ == '__main__':
    sys.exit(main())
