import json,pickle,random,collections as C,sys
S=sys.argv[1]
rows=pickle.load(open(S+'/rows.pkl','rb'))
D={"pull-up-account":"Look up the customer's account by full name or account ID","verify-identity":"Verify identity with name/account ID/order ID/zip code","search-faq":"Open the FAQ/policy search for a general question","select-faq":"Choose the FAQ answer to read out","validate-purchase":"Check a purchase using username, email and order ID","enter-details":"Enter extra details (address, username, email, PIN...) in the details form","ask-the-oracle":"Query the system oracle whether the customer's claim/request is valid","record-reason":"Record the reason/refund target/payment method etc. from the customer","update-order":"Change an order (address, date, item, method, price)","membership":"Check membership level privileges","shipping-status":"Look up shipping status/options","log-out-in":"Troubleshooting step: ask customer to log out and in","offer-refund":"Offer a refund with a dollar amount","send-link":"Send a reset/security/info link to the customer","update-account":"Add/remove/renew/extend a service on the account","try-again":"Troubleshooting step: ask customer to try again","subscription-status":"Look up subscription status, amount due or due date","notify-team":"Notify manager, website team or purchasing department","promo-code":"Issue a promo code","make-purchase":"Add a product to an order","search-membership":"Search FAQ: membership questions","search-pricing":"Search FAQ: pricing questions","instructions":"Give troubleshooting instructions","search-policy":"Search FAQ: policy questions","make-password":"Generate a new password","search-timing":"Search FAQ: timing questions","search-boots":"Search FAQ: boots","search-jeans":"Search FAQ: jeans","search-jacket":"Search FAQ: jacket","search-shirt":"Search FAQ: shirt"}
assert len(D)==30
def row(cid,sp,dd,fl,only_actions=True):
    opts=dict(D)
    if not only_actions: opts["respond"]="Reply to the customer in natural language (no button)"
    return {"decision_type":"next_button_after_"+dd['last'] if only_actions else "next_step_after_"+dd['last'],
      "question":"A customer-service agent follows company guidelines and is about to press a button on the agent dashboard. Which button do they press next?" if only_actions else "A customer-service agent follows company guidelines. What is their next step: a dashboard button, or a natural-language reply?",
      "options":opts,"context":{"dialogue":dd['ctx'] or "(conversation just started)"},
      "label":dd['label'],"meta":{"task_id":cid,"split":sp,"flow":fl,"subflow":dd['sub'],"turn_count":dd['idx']}}
random.seed(7)
act=[r for r in rows if r[2]['label']!='respond']
ex=[]
for last in ['none','verify-identity','ask-the-oracle']:
    ex.append(random.choice([r for r in act if r[2]['last']==last and len(r[2]['ctx'])>300]))
with open(S+'/examples.jsonl','w') as f:
    for r in ex: f.write(json.dumps(row(*r),ensure_ascii=False)+'\n')
# cost
sample=row(*ex[0]); fixed=len(json.dumps({k:sample[k] for k in ('question','options')},ensure_ascii=False))
fixed31=fixed+len('"respond": "Reply to the customer in natural language (no button)", ')
print('fixed chars',fixed,fixed31)
ca=[len(r[2]['ctx']) for r in act]; cb=[len(r[2]['ctx']) for r in rows]
def cost(n,ctxmean,fx): 
    tok=n*(ctxmean+fx)/3.5; return tok/1e6, tok/1e6*0.042
print('action-only full', len(act), cost(len(act),sum(ca)/len(ca),fixed))
print('action-only 5k', cost(5000,sum(ca)/len(ca),fixed))
print('all-step full', len(rows), cost(len(rows),sum(cb)/len(cb),fixed31))
print('all-step 5k', cost(5000,sum(cb)/len(cb),fixed31))
# sessions in 5k subsample if random decisions vs one-per-session
# decision types with <80 sessions (action-only)
bt=C.defaultdict(set)
for r in act: bt[r[2]['last']].add(r[0])
print([ (k,len(v)) for k,v in bt.items() if len(v)<80])
# sessions in test split etc
print(C.Counter(r[1] for r in act))
