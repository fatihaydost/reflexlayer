import json
D='bfcl/repo/berkeley-function-call-leaderboard/bfcl_eval/data/'
out=open('../bfcl_decisions.jsonl','w'); tot=0; n=0; cnt={}
for name in ['live_multiple','multiple']:
    gold={json.loads(l)['id']:json.loads(l)['ground_truth'] for l in open(D+f'possible_answer/BFCL_v4_{name}.json')}
    for l in open(D+f'BFCL_v4_{name}.json'):
        r=json.loads(l); lab=list(gold[r['id']][0].keys())[0]
        opts={f['name']:f.get('description','')[:300] for f in r['function']}
        assert lab in opts
        user=r['question'][0][-1]['content']
        row={"decision_type":f"bfcl_{name}","question":"Which function should be called to satisfy the user's request?","options":opts,"context":{"user_request":user[:3000]},"label":lab}
        out.write(json.dumps(row)+'\n'); c=len(json.dumps(opts))+len(user); tot+=c; n+=1; cnt[name]=cnt.get(name,0)+1
print(n,cnt,'avg chars',tot//n,'total',tot)
