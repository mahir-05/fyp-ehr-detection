from sqlalchemy import create_engine
import pandas as pd, numpy as np, json, re, warnings
from scipy.stats import chi2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from datetime import timedelta
warnings.filterwarnings('ignore')

engine = create_engine("mysql+mysqlconnector://openemr:openemrpass@localhost:3306/openemr")
print("="*70)
print("McNEMAR'S TEST v3 - CORRECT METHODOLOGY")
print("Rules on FULL data, GB trained on 80%, compare on 20% test set")
print("="*70)

# load the simulation data and parse ground truth labels from the comments field
print("\n[1/7] Loading data...")
df = pd.read_sql("SELECT id,date,event,user,patient_id,success,comments FROM log WHERE comments LIKE 'SIMULATED|%%' ORDER BY date", engine)
def parse_gt(c):
    try:
        if c and 'SIMULATED|' in c: return json.loads(c.replace('SIMULATED|',''))
    except: pass
    return {'is_malicious':False}
df['is_malicious'] = df['comments'].apply(lambda c: parse_gt(c).get('is_malicious',False))
df['date'] = pd.to_datetime(df['date'])
patients_df = pd.read_sql("SELECT pid,fname,lname,postal_code,street FROM patient_data WHERE pid>0", engine)
users_df = pd.read_sql("SELECT id,username,fname,lname,title FROM users WHERE username!=''", engine)
y = df['is_malicious'].astype(int).values
print(f"    {len(df)} entries ({y.sum()} malicious, {len(y)-y.sum()} legitimate)")

# same 80/20 stratified split as the ML comparison script so the test sets line up
print("\n[2/7] Creating 80/20 stratified split (random_state=42)...")
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
y_test = y[test_idx]
test_ids = set(df.iloc[test_idx]['id'].tolist())
print(f"    Train: {len(train_idx)} | Test: {len(test_idx)} | Test malicious: {y_test.sum()}")

# rules need to run on the full dataset because context like hourly volume only
# makes sense across the whole log - then I just pull the test set predictions out
print("\n[3/7] Rule-based on FULL dataset (rules need full context)...")
TH = {'normal_hours_start':6,'normal_hours_end':22,'max_patients_per_hour':10,
      'sequential_window_minutes':5,'sequential_min_length':5,'same_patient_access_threshold':5,
      'extended_viewing_window_hours':2,'failed_access_threshold':3,
      'same_surname_threshold':2,'same_surname_time_window_minutes':45,'same_surname_min_gap_seconds':30,
      'same_location_threshold':5,'geographic_time_window_hours':2,
      'restricted_roles':['Receptionist'],'cross_dept_min_accesses':6,'cross_dept_time_window_hours':3,
      'vip_patient_ids':[103,110,69],'vip_only_flag_users':['malicious_user'],
      'street_match_threshold':2,'street_match_time_window_hours':4}
fids = set()
sl = patients_df.set_index('pid')['lname'].to_dict()
pl = patients_df.set_index('pid')['postal_code'].to_dict()
rl = users_df.set_index('username')['title'].to_dict()

# R1: flag anything outside 6am-10pm
for _,r in df.iterrows():
    h=r['date'].hour
    if h<TH['normal_hours_start'] or h>=TH['normal_hours_end']: fids.add(r['id'])

# R2: more than 10 distinct patients in one hour
dt=df.copy(); dt['hb']=dt['date'].dt.floor('h')
for (_,_),g in dt.groupby(['user','hb']):
    if g['patient_id'].nunique()>TH['max_patients_per_hour']: fids.update(g['id'].tolist())

# R3: alphabetical browsing - accessing patients A-Z in sequence
for u,ud in df.groupby('user'):
    ud=ud.sort_values('date'); seq=[]; lt=None
    for _,r in ud.iterrows():
        sn=sl.get(r['patient_id'],''); ct=r['date']
        if lt is None or (ct-lt).total_seconds()/60<=TH['sequential_window_minutes']:
            seq.append({'id':r['id'],'s':sn,'t':ct})
        else:
            if len(seq)>=TH['sequential_min_length']:
                ss=[s['s'] for s in seq]
                if ss==sorted(ss): fids.update(s['id'] for s in seq)
            seq=[{'id':r['id'],'s':sn,'t':ct}]
        lt=ct
    if len(seq)>=TH['sequential_min_length']:
        ss=[s['s'] for s in seq]
        if ss==sorted(ss): fids.update(s['id'] for s in seq)

# R4: multiple patients with the same surname accessed in a short window
dt2=df.copy(); dt2['ps']=dt2['patient_id'].map(sl); fsids=set()
for u,ud in dt2.groupby('user'):
    ud=ud.sort_values('date')
    for _,r in ud.iterrows():
        we=r['date']+timedelta(minutes=TH['same_surname_time_window_minutes'])
        wd=ud[(ud['date']>=r['date'])&(ud['date']<=we)]
        for sn in wd['ps'].dropna().unique():
            sd=wd[wd['ps']==sn]
            if sd['patient_id'].nunique()>=TH['same_surname_threshold']:
                ts=sd['date'].sort_values()
                if len(ts)>=2 and any(g>=timedelta(seconds=TH['same_surname_min_gap_seconds']) for g in ts.diff().dropna()):
                    fsids.update(sd['id'].tolist())
fids.update(fsids)

# R5: accessing lots of patients from the same postcode - neighbour snooping
dt2['pc']=df['patient_id'].map(pl); gids=set()
for u,ud in dt2.groupby('user'):
    ud=ud.sort_values('date')
    for _,r in ud.iterrows():
        we=r['date']+timedelta(hours=TH['geographic_time_window_hours'])
        wd=ud[(ud['date']>=r['date'])&(ud['date']<=we)]
        for pc in wd['pc'].dropna().unique():
            pd2=wd[wd['pc']==pc]
            if pd2['patient_id'].nunique()>=TH['same_location_threshold']: gids.update(pd2['id'].tolist())
fids.update(gids)

# R6: restricted staff (receptionist) accessing clinical records they shouldn't need
skw=['diagnosis','medication','lab','clinical','notes']
cdf=df[df['event'].apply(lambda e:any(k in str(e).lower() for k in skw) if e else False)]
xids=set()
for u,ud in cdf.groupby('user'):
    ur=rl.get(u,'') or ''
    if not any(r.lower() in ur.lower() for r in TH['restricted_roles']): continue
    ud=ud.sort_values('date')
    for _,r in ud.iterrows():
        we=r['date']+timedelta(hours=TH['cross_dept_time_window_hours'])
        wd=ud[(ud['date']>=r['date'])&(ud['date']<=we)]
        if len(wd)>=TH['cross_dept_min_accesses']: xids.update(wd['id'].tolist())
fids.update(xids)

# R7: same patient accessed 5+ times in a 2-hour window
dt2['tb']=dt2['date'].dt.floor(f'{TH["extended_viewing_window_hours"]}h')
for (u,p,b),g in dt2.groupby(['user','patient_id','tb']):
    if len(g)>TH['same_patient_access_threshold']: fids.update(g['id'].tolist())

# R8: 3+ failed/denied access attempts
fdf=df[df['event'].str.contains('denied|failed|unauthorized',case=False,na=False)]
for u,g in fdf.groupby('user'):
    if len(g)>=TH['failed_access_threshold']: fids.update(g['id'].tolist())

# R9: flagged user accessing a VIP patient
vdf=df[df['patient_id'].isin(TH['vip_patient_ids'])]
for _,r in vdf.iterrows():
    if r['user'] in TH['vip_only_flag_users']: fids.add(r['id'])

# R10: street-level clustering, even more specific than postcode
def ns(s):
    if not s or pd.isna(s): return None
    s=str(s).lower().strip()
    s=re.sub(r'\b(street|st|avenue|ave|road|rd|drive|dr|lane|ln|court|ct|place|pl|boulevard|blvd)\b','',s)
    s=re.sub(r'[^a-z0-9\s]','',s); s=re.sub(r'\s+',' ',s).strip()
    return s if len(s)>3 else None
stl={p:ns(s) for p,s in patients_df.set_index('pid')['street'].to_dict().items()}
dt2['sn']=df['patient_id'].map(stl); sids=set()
for u,ud in dt2.groupby('user'):
    ud=ud.sort_values('date')
    for _,r in ud.iterrows():
        if not r['sn']: continue
        we=r['date']+timedelta(hours=TH['street_match_time_window_hours'])
        wd=ud[(ud['date']>=r['date'])&(ud['date']<=we)&(ud['sn'].notna())]
        for st in wd['sn'].unique():
            if not st: continue
            sd=wd[wd['sn']==st]
            if sd['patient_id'].nunique()>=TH['street_match_threshold']: sids.update(sd['id'].tolist())
fids.update(sids)

# now pull just the predictions for the test set entries
all_ids = df['id'].values
rb_all = np.array([1 if aid in fids else 0 for aid in all_ids])
rb_preds = rb_all[test_idx]
rb_f1=f1_score(y_test,rb_preds)*100; rb_acc=accuracy_score(y_test,rb_preds)*100
print(f"    Rule-Based (test subset): Acc={rb_acc:.2f}%, F1={rb_f1:.2f}%")

# train gradient boosting on the 80% training portion, then predict on the test set
print("\n[4/7] Gradient Boosting: train 80%, predict test...")
feat=pd.DataFrame(index=df.index)
feat['hour']=df['date'].dt.hour
feat['is_after_hours']=((df['date'].dt.hour<6)|(df['date'].dt.hour>=22)).astype(int)
feat['is_night']=((df['date'].dt.hour>=22)|(df['date'].dt.hour<=5)).astype(int)
feat['is_weekend']=(df['date'].dt.dayofweek>=5).astype(int)
feat['day_of_week']=df['date'].dt.dayofweek
feat['user_encoded']=LabelEncoder().fit_transform(df['user'].fillna('unknown'))
ur2=df['user'].map(rl).fillna('unknown')
feat['is_receptionist']=ur2.str.lower().str.contains('receptionist',na=False).astype(int)
feat['event_encoded']=LabelEncoder().fit_transform(df['event'].fillna('unknown'))
feat['is_clinical_event']=df['event'].fillna('').str.lower().apply(lambda x:any(k in x for k in skw)).astype(int)
uc=df.groupby('user').size(); feat['user_total_accesses']=df['user'].map(uc)
dft=df.copy(); dft['hb']=dft['date'].dt.floor('h')
hr=dft.groupby(['user','hb']).size().reset_index(name='hc'); dft=dft.merge(hr,on=['user','hb'],how='left')
feat['hourly_access_count']=dft['hc'].values
dft['surname']=df['patient_id'].map(sl)
sc=dft.groupby(['user','surname']).size().reset_index(name='sc2'); dft=dft.merge(sc,on=['user','surname'],how='left')
feat['same_surname_count']=dft['sc2'].fillna(1).values
dft['postal']=df['patient_id'].map(pl)
pc2=dft.groupby(['user','postal']).size().reset_index(name='pc3'); dft=dft.merge(pc2,on=['user','postal'],how='left')
feat['same_postal_count']=dft['pc3'].fillna(1).values
feat['is_denied_event']=df['event'].fillna('').str.lower().str.contains('denied|failed',na=False).astype(int)
X=feat.fillna(0)
gb=GradientBoostingClassifier(n_estimators=100,max_depth=5,random_state=42)
gb.fit(X.iloc[train_idx],y[train_idx])
gb_preds=gb.predict(X.iloc[test_idx])
gb_f1=f1_score(y_test,gb_preds)*100; gb_acc=accuracy_score(y_test,gb_preds)*100
print(f"    Gradient Boosting: Acc={gb_acc:.2f}%, F1={gb_f1:.2f}%")

# McNemar's is the right test here because both classifiers see the exact same examples
# a standard t-test would be wrong since the predictions aren't independent
# using continuity correction (the -1) as recommended by Dietterich (1998)
print("\n[5/7] McNemar's Test...")
print("-"*70)
n=len(y_test)
rc=(rb_preds==y_test); gc=(gb_preds==y_test)
a=int(np.sum(rc&gc)); b=int(np.sum(rc&~gc)); c=int(np.sum(~rc&gc)); d=int(np.sum(~rc&~gc))
print(f"\n    Contingency Table (n={n}):")
print(f"    {'':>25} {'GB Correct':>12} {'GB Wrong':>12} {'Total':>8}")
print(f"    {'-'*60}")
print(f"    {'Rule Correct':<25} {a:>12} {b:>12} {a+b:>8}")
print(f"    {'Rule Wrong':<25} {c:>12} {d:>12} {c+d:>8}")
print(f"    {'-'*60}")
print(f"    {'Total':<25} {a+c:>12} {b+d:>12} {n:>8}")
print(f"\n    b={b}, c={c}, disagreements={b+c}")

if (b+c)==0: chi2_stat=0.0; p_value=1.0
elif (b+c)<25:
    # when b+c is small the chi-squared approximation breaks down, use exact binomial instead
    # binomtest is the newer scipy API, binom_test is the fallback for older versions
    chi2_stat=(abs(b-c)-1)**2/(b+c)
    try:
        from scipy.stats import binomtest; p_value=binomtest(b,b+c,0.5).pvalue
    except:
        from scipy.stats import binom_test; p_value=binom_test(b,b+c,0.5)
    print(f"    Note: b+c={b+c} < 25, exact binomial test used")
else:
    chi2_stat=(abs(b-c)-1)**2/(b+c); p_value=1-chi2.cdf(chi2_stat,df=1)

sig=p_value<0.05
print(f"\n    chi2={chi2_stat:.4f}, p={p_value:.4f}")
if not sig:
    print(f"    RESULT: p={p_value:.4f} > 0.05 -> NOT significant")
else:
    print(f"    RESULT: p={p_value:.4f} < 0.05 -> SIGNIFICANT")

# keep terminal output concise - don't print the full dissertation text block
print(f"\n[6/7] Preparing summary outputs...")
print("    Dissertation text not printed to terminal")

# save the key numbers to CSV for reference
print("[7/7] Saving...")
pd.DataFrame([{'test_size':n,'a':a,'b':b,'c':c,'d':d,'chi2':round(chi2_stat,4),
    'p_value':round(p_value,4),'significant':sig,'rb_f1':round(rb_f1,2),
    'gb_f1':round(gb_f1,2),'gap':round(gb_f1-rb_f1,2)}]).to_csv('outputs/mcnemars_test_results.csv',index=False)
print("    Saved: outputs/mcnemars_test_results.csv")
with open('outputs/mcnemars_dissertation_text.txt','w') as f:
    f.write(f"chi2={chi2_stat:.4f}, p={p_value:.4f}, b={b}, c={c}, significant={sig}")
print("    Saved: outputs/mcnemars_dissertation_text.txt")
