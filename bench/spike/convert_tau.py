import json,re,glob,os,collections
R='tau-bench/tau_bench/envs/'
def tooldesc(dom):
    d={}
    for f in glob.glob(R+dom+'/tools/*.py'):
        s=open(f).read()
        n=re.search(r'"name":\s*"(\w+)"',s); m=re.search(r'"description":\s*(\(?\s*(?:"[^"\n]*"\s*)+)',s)
        if n and m: d[n.group(1)]=''.join(re.findall(r'"([^"\n]*)"',m.group(1)))[:300]
    d['respond']='Send a message to the user (ask a question, answer, confirm) instead of calling a tool.'
    return d
CAP=5000
out=open('../tau_bench_decisions.jsonl','w'); stats=collections.defaultdict(list)
for dom in ['retail','airline']:
    td=tooldesc(dom)
    for f in glob.glob(f'tau-bench/historical_trajectories/*{dom}.json'):
        for x in json.load(open(f)):
            if x['reward']!=1: continue
            tr=x['traj']
            for i,m in enumerate(tr):
                if m['role']!='assistant': continue
                tc=m.get('tool_calls'); lab=tc[0]['function']['name'] if tc else 'respond'
                turns=[]
                for p in tr[1:i]:
                    if p['role']=='tool': turns.append(f"[tool result {p['name']}] {p['content'][:600]}")
                    elif p['role']=='assistant' and p.get('tool_calls'): turns.append("[agent called] "+', '.join(c['function']['name']+str(c['function']['arguments']) for c in p['tool_calls']))
                    else: turns.append(f"[{p['role']}] {p['content']}")
                ctx='\n'.join(turns)[-CAP:]
                row={"decision_type":f"tau_{dom}_next_action","question":"Given the conversation so far, what should the customer-service agent do next: call one tool or reply to the user?","options":{k:v for k,v in td.items()},"context":{"conversation_tail":ctx},"label":lab}
                out.write(json.dumps(row)+'\n'); stats[dom].append(len(ctx)+len(json.dumps(row['options'])))
for k,v in stats.items(): print(k,len(v),'avg chars (ctx+options)',sum(v)//len(v))
