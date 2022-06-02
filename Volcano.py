from Visualizations import *

sns.set_theme(style="whitegrid")

# Reading data
files = ['2_Lipidomics_Both_Annotated/control_white vs. control_cortex/control_white vs. control_cortex.csv',
         '2_Lipidomics_Both_Annotated/frontal_cortex_control vs. temporal_cortex_control/frontal_cortex_control vs. temporal_cortex_control.csv',
         '2_Lipidomics_Both_Annotated/frontal_white_control vs. temporal_white_control/frontal_white_control vs. temporal_white_control.csv',
         '2_Lipidomics_Both_Annotated/schizophrenia vs. control/schizophrenia vs. control.csv',
         '2_Lipidomics_Both_Annotated/schizo_white vs. control_white/schizo_white vs. control_white.csv',
         '2_Lipidomics_Both_Annotated/schizo_cortex vs. control_cortex/schizo_cortex vs. control_cortex.csv',
         '2_Lipidomics_Both_Annotated/schizo_temporal vs. control_temporal/schizo_temporal vs. control_temporal.csv',
         '2_Lipidomics_Both_Annotated/schizo_frontal vs. control_frontal/schizo_frontal vs. control_frontal.csv']

filenames = []
for element in files:
    filenames.append((element.partition('/')[2].partition('/')[2])[:-4])

for file, filename in zip(files, filenames):
    volcano_ann(file, filename)
