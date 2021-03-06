import pandas as pd

vec_1 = pd.read_csv('./latent_space/logPvals/p_val_0to300.txt')
vec_2 = pd.read_csv('./latent_space/logPvals/p_val_300to600.txt')
vec_3 = pd.read_csv('./latent_space/logPvals/p_val_600to900.txt')
vec_4 = pd.read_csv('./latent_space/logPvals/p_val_900to1200.txt')
vec_5 = pd.read_csv('./latent_space/logPvals/p_val_1200to1500.txt')
vec_6 = pd.read_csv('./latent_space/logPvals/p_val_1500to1800.txt')
vec_7 = pd.read_csv('./latent_space/logPvals/p_val_1800to2100.txt')
vec_8 = pd.read_csv('./latent_space/logPvals/p_val_2100to2400.txt')
vec_9 = pd.read_csv('./latent_space/logPvals/p_val_2400to2700.txt')
vec_10 = pd.read_csv('./latent_space/logPvals/p_val_2700to3000.txt')
vec_11 = pd.read_csv('./latent_space/logPvals/p_val_3000to3300.txt')
vec_12 = pd.read_csv('./latent_space/logPvals/p_val_3300to3600.txt')
vec_13 = pd.read_csv('./latent_space/logPvals/p_val_3600to3900.txt')
vec_14 = pd.read_csv('./latent_space/logPvals/p_val_3900to4200.txt')
vec_15 = pd.read_csv('./latent_space/logPvals/p_val_4200to4500.txt')
vec_16 = pd.read_csv('./latent_space/logPvals/p_val_4500to4800.txt')
vec_17 = pd.read_csv('./latent_space/logPvals/p_val_4800to5100.txt')
vec_18 = pd.read_csv('./latent_space/logPvals/p_val_5100to5400.txt')
vec_19 = pd.read_csv('./latent_space/logPvals/p_val_5400to5700.txt')
vec_20 = pd.read_csv('./latent_space/logPvals/p_val_5700to6000.txt')
vec_21 = pd.read_csv('./latent_space/logPvals/p_val_6000to6300.txt')
vec_22 = pd.read_csv('./latent_space/logPvals/p_val_6300to6600.txt')
vec_23 = pd.read_csv('./latent_space/logPvals/p_val_6600to6900.txt')
vec_24 = pd.read_csv('./latent_space/logPvals/p_val_6900to7200.txt')
vec_25 = pd.read_csv('./latent_space/logPvals/p_val_7200to7500.txt')
vec_26 = pd.read_csv('./latent_space/logPvals/p_val_7500to7800.txt')
vec_27 = pd.read_csv('./latent_space/logPvals/p_val_7800to8100.txt')
vec_28 = pd.read_csv('./latent_space/logPvals/p_val_8100to8400.txt')
vec_29 = pd.read_csv('./latent_space/logPvals/p_val_8400to8700.txt')
vec_30 = pd.read_csv('./latent_space/logPvals/p_val_8700to9000.txt')
vec_31 = pd.read_csv('./latent_space/logPvals/p_val_9000to9300.txt')
vec_32 = pd.read_csv('./latent_space/logPvals/p_val_9300to9600.txt')
vec_33 = pd.read_csv('./latent_space/logPvals/p_val_9600to9900.txt')
vec_34 = pd.read_csv('./latent_space/logPvals/p_val_9900to10000.txt')

p_vals = pd.concat([vec_1, vec_2], ignore_index=True)
p_vals = pd.concat([p_vals, vec_3], ignore_index=True)
p_vals = pd.concat([p_vals, vec_4], ignore_index=True)
p_vals = pd.concat([p_vals, vec_5], ignore_index=True)
p_vals = pd.concat([p_vals, vec_6], ignore_index=True)
p_vals = pd.concat([p_vals, vec_7], ignore_index=True)
p_vals = pd.concat([p_vals, vec_8], ignore_index=True)
p_vals = pd.concat([p_vals, vec_9], ignore_index=True)
p_vals = pd.concat([p_vals, vec_10], ignore_index=True)
p_vals = pd.concat([p_vals, vec_11], ignore_index=True)
p_vals = pd.concat([p_vals, vec_12], ignore_index=True)
p_vals = pd.concat([p_vals, vec_13], ignore_index=True)
p_vals = pd.concat([p_vals, vec_14], ignore_index=True)
p_vals = pd.concat([p_vals, vec_15], ignore_index=True)
p_vals = pd.concat([p_vals, vec_16], ignore_index=True)
p_vals = pd.concat([p_vals, vec_17], ignore_index=True)
p_vals = pd.concat([p_vals, vec_18], ignore_index=True)
p_vals = pd.concat([p_vals, vec_19], ignore_index=True)
p_vals = pd.concat([p_vals, vec_20], ignore_index=True)
p_vals = pd.concat([p_vals, vec_21], ignore_index=True)
p_vals = pd.concat([p_vals, vec_22], ignore_index=True)
p_vals = pd.concat([p_vals, vec_23], ignore_index=True)
p_vals = pd.concat([p_vals, vec_24], ignore_index=True)
p_vals = pd.concat([p_vals, vec_25], ignore_index=True)
p_vals = pd.concat([p_vals, vec_26], ignore_index=True)
p_vals = pd.concat([p_vals, vec_27], ignore_index=True)
p_vals = pd.concat([p_vals, vec_28], ignore_index=True)
p_vals = pd.concat([p_vals, vec_29], ignore_index=True)
p_vals = pd.concat([p_vals, vec_30], ignore_index=True)
p_vals = pd.concat([p_vals, vec_31], ignore_index=True)
p_vals = pd.concat([p_vals, vec_32], ignore_index=True)
p_vals = pd.concat([p_vals, vec_33], ignore_index=True)
p_vals = pd.concat([p_vals, vec_34], ignore_index=True).drop(columns={'Unnamed: 0'})



inp_1 = pd.read_csv('./latent_space/encodedZINC_0to300_mean.txt')
inp_2 = pd.read_csv('./latent_space/encodedZINC_300to600_mean.txt')
inp_3 = pd.read_csv('./latent_space/encodedZINC_600to900_mean.txt')
inp_4 = pd.read_csv('./latent_space/encodedZINC_900to1200_mean.txt')
inp_5 = pd.read_csv('./latent_space/encodedZINC_1200to1500_mean.txt')
inp_6 = pd.read_csv('./latent_space/encodedZINC_1500to1800_mean.txt')
inp_7 = pd.read_csv('./latent_space/encodedZINC_1800to2100_mean.txt')
inp_8 = pd.read_csv('./latent_space/encodedZINC_2100to2400_mean.txt')
inp_9 = pd.read_csv('./latent_space/encodedZINC_2400to2700_mean.txt')
inp_10 = pd.read_csv('./latent_space/encodedZINC_2700to3000_mean.txt')
inp_11 = pd.read_csv('./latent_space/encodedZINC_3000to3300_mean.txt')
inp_12 = pd.read_csv('./latent_space/encodedZINC_3300to3600_mean.txt')
inp_13 = pd.read_csv('./latent_space/encodedZINC_3600to3900_mean.txt')
inp_14 = pd.read_csv('./latent_space/encodedZINC_3900to4200_mean.txt')
inp_15 = pd.read_csv('./latent_space/encodedZINC_4200to4500_mean.txt')
inp_16 = pd.read_csv('./latent_space/encodedZINC_4500to4800_mean.txt')
inp_17 = pd.read_csv('./latent_space/encodedZINC_4800to5100_mean.txt')
inp_18 = pd.read_csv('./latent_space/encodedZINC_5100to5400_mean.txt')
inp_19 = pd.read_csv('./latent_space/encodedZINC_5400to5700_mean.txt')
inp_20 = pd.read_csv('./latent_space/encodedZINC_5700to6000_mean.txt')
inp_21 = pd.read_csv('./latent_space/encodedZINC_6000to6300_mean.txt')
inp_22 = pd.read_csv('./latent_space/encodedZINC_6300to6600_mean.txt')
inp_23 = pd.read_csv('./latent_space/encodedZINC_6600to6900_mean.txt')
inp_24 = pd.read_csv('./latent_space/encodedZINC_6900to7200_mean.txt')
inp_25 = pd.read_csv('./latent_space/encodedZINC_7200to7500_mean.txt')
inp_26 = pd.read_csv('./latent_space/encodedZINC_7500to7800_mean.txt')
inp_27 = pd.read_csv('./latent_space/encodedZINC_7800to8100_mean.txt')
inp_28 = pd.read_csv('./latent_space/encodedZINC_8100to8400_mean.txt')
inp_29 = pd.read_csv('./latent_space/encodedZINC_8400to8700_mean.txt')
inp_30 = pd.read_csv('./latent_space/encodedZINC_8700to9000_mean.txt')
inp_31 = pd.read_csv('./latent_space/encodedZINC_9000to9300_mean.txt')
inp_32 = pd.read_csv('./latent_space/encodedZINC_9300to9600_mean.txt')
inp_33 = pd.read_csv('./latent_space/encodedZINC_9600to9900_mean.txt')
inp_34 = pd.read_csv('./latent_space/encodedZINC_9900to10000_mean.txt')

encoded_df = pd.concat([inp_1, inp_2], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_3], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_4], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_5], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_6], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_7], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_8], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_9], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_10], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_11], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_12], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_13], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_14], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_15], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_16], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_17], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_18], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_19], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_20], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_21], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_22], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_23], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_24], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_25], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_26], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_27], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_28], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_29], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_30], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_31], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_32], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_33], ignore_index=True)
encoded_df = pd.concat([encoded_df, inp_34], ignore_index=True).drop(columns={'Unnamed: 0'})

encoded_df.to_csv('./latent_space/encoded_ZINC_mean.txt')