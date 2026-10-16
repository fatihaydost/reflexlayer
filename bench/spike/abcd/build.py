import json,collections as C,sys,statistics as st,random
S=sys.argv[1]
d=json.load(open(S+'/data/abcd.json'))
def fmt(o,x,prev_actions=None):
    sp=x['speaker']; t=o[1]
    if sp=='action':
        return f"[system] {t}"
    return f"{sp.capitalize()}: {t}"
def decisions(c):
    out=[];lines=[];last='none';nact=0
    for o,x in zip(c['original'],c['delexed']):
        tg=x['targets']
        if x['speaker']!='customer':
            lab=tg[2] if tg[1]=='take_action' else 'respond'
            out.append(dict(ctx='\n'.join(lines),label=lab,last=last,idx=x['turn_count'],sub=tg[0],vals=tg[3]))
        if x['speaker']=='action':
            a=tg[2]; v=', '.join(tg[3])
            lines.append(f"[Agent pressed button: {a}({v})] -> {o[1]}"); last=a; nact+=1
        else: lines.append(fmt(o,x))
    return out
tot=C.Counter(); bytype=C.defaultdict(lambda:[set(),C.Counter()]); chars=[]; achars=[]
n_over=0; actionsess=0
allrows=[]
for sp,s in d.items():
    for c in s:
        ds=decisions(c); hasact=False
        for dd in ds:
            L=len(dd['ctx']); chars.append(L)
            if dd['label']!='respond': achars.append(L); hasact=True
            bt=bytype[dd['last']]; bt[0].add(c['convo_id']); bt[1][dd['label']]+=1
            allrows.append((c['convo_id'],sp,dd,c['scenario']['flow']))
        actionsess+=hasact
print('decisions',len(chars),'actions',len(achars),'sessions with action',actionsess)
print('ctx chars all mean/median/p95/max',st.mean(chars),st.median(chars),sorted(chars)[int(.95*len(chars))],max(chars),' over6000',sum(1 for x in chars if x>6000))
print('action ctx mean',st.mean(achars),st.median(achars),sorted(achars)[int(.95*len(achars))])
print('--- by last-action type: sessions, decisions, nlabels, majority share, action-share')
for k,(ses,cn) in sorted(bytype.items(),key=lambda kv:-len(kv[1][0])):
    n=sum(cn.values()); m=cn.most_common(1)[0]
    # majority among non-respond
    print(f"{k:22s} sess={len(ses):5d} dec={n:6d} labels={len(cn):2d} maj={m[0]}:{m[1]/n:.2f}")
json.dump([[a,b,c,e] for a,b,c,e in allrows[:0]],open(S+'/x.json','w'))
# examples
import pickle
random.seed(1)
pickle.dump(allrows,open(S+'/rows.pkl','wb'))
