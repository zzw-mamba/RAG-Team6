from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from raglab.data import iter_records

PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG Critic Lab · 过程回放</title>
<style>
body{font:16px/1.6 system-ui;margin:0;background:#f4f6fa;color:#192331}
main{max-width:1000px;margin:40px auto;padding:0 20px}h1{margin-bottom:6px}
.note{color:#526176}select,input{padding:12px;width:100%;box-sizing:border-box;margin:8px 0}
.card{background:white;border:1px solid #dbe1eb;border-radius:12px;padding:20px;margin:16px 0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 ui-monospace,monospace}
.tag{background:#e8efff;padding:4px 9px;border-radius:8px;display:inline-block;margin-right:8px}
details{margin:8px 0}.final{border-left:5px solid #20645a}
</style><main><h1>RAG Critic Lab</h1><div class="note">离线过程回放 · 新问题先运行 infer 生成轨迹</div>
<input id="search" placeholder="输入问题关键词筛选"><select id="pick" aria-label="选择问题"></select>
<div id="result"></div></main><script>
let traces=[];const pick=document.getElementById('pick'),root=document.getElementById('result');
function text(tag,value,parent){let e=document.createElement(tag);e.textContent=value;parent.append(e);return e}
function card(title,value,cls=''){let c=document.createElement('section');c.className='card '+cls;
text('h2',title,c);text('pre',value,c);root.append(c);return c}
function render(){root.replaceChildren();let t=traces[Number(pick.value)];if(!t)return;
let h=card('问题',t.question);text('span',t.mode+' · '+t.backend,h).className='tag';
text('span',t.stop_reason+' · 修正 '+t.revisions+' 次 · '+t.elapsed_seconds.toFixed(3)+' 秒',h).className='tag';
if(t.backend.split('+').includes('mock'))text('p','合成 Mock 演示，不能作为真实模型实验结果。',h);
let docs=card('检索文档','');for(let d of t.documents){let el=document.createElement('details');
text('summary',d.id+' · '+d.title,el);text('pre',d.text,el);docs.append(el)}
for(let s of t.steps){if(s.kind==='retrieve')continue;card(s.index+' · '+s.kind,JSON.stringify(s.content,null,2))}
card('最终答案',t.final_answer,'final')}
function filter(){let q=document.getElementById('search').value.toLocaleLowerCase();pick.replaceChildren();
traces.forEach((t,i)=>{if(t.question.toLocaleLowerCase().includes(q)){let o=document.createElement('option');
o.value=String(i);o.textContent=t.id+' · '+t.question;pick.append(o)}});render()}
pick.onchange=render;document.getElementById('search').oninput=filter;
fetch('/traces').then(r=>r.json()).then(data=>{traces=data;filter()});
</script></html>"""


def serve(run_dir: Path, port: int = 8765) -> None:
    traces = json.dumps(list(iter_records(run_dir / "traces.jsonl")), ensure_ascii=False).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                body, content_type = PAGE.encode(), "text/html; charset=utf-8"
            elif self.path == "/traces":
                body, content_type = traces, "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Trace replay: http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
