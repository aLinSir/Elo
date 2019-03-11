import pandas as pd


n = 3
A = pd.read_csv('submissions/Bestoutput.csv')
B = pd.read_csv('submissions/lgb2.csv')
C = pd.read_csv('submissions/submission.csv')
D = pd.read_csv('submissions/blending1.csv')
E = pd.read_csv('submissions/xgb.csv')
F = pd.read_csv('submissions/lgb1.csv')
















predictions = 0.164*A['target'] + 0.166*B['target'] + 0.166*C['target'] + 0.172*D['target'] + 0.166*E['target'] + 0.166*F['target']
# predictions = final / n
df = pd.DataFrame({'card_id': A['card_id'],
                   'target': predictions})
df.to_csv('submissions/final.csv', index=False)