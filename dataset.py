import pandas as pd
import os
import shutil

df = pd.read_excel("C:\\Users\jones\Desktop\MammData\All FFDM Case "
                    "Truth Summary-ROI-MassCenter.xlsx", sheet_name='Mass-Center-Marks')

### To create an excel file with only the CC view
# onlyCC = pd.DataFrame()
# for i in range(len(df)):
#     if df['image_name'][i][-6:][:2] == 'CC':
#         print("This is a CC view")
#         onlyCC = onlyCC.append(df.iloc[i])

ccOnly = pd.read_excel("onlyCC.xlsx", sheet_name='clean')

## To copy the images into its own directory
databases = ["C:\\Users\jones\Desktop\MammData\FFDM_Database-1",
             "C:\\Users\jones\Desktop\MammData\FFDM_Database-2",
             "C:\\Users\jones\Desktop\MammData\FFDM_Database-3"]
dest = "C:\\Users\jones\Desktop\MammData\ccOnly"
count = 0
# for i in range(len(databases)):
#     for filename in os.listdir(databases[i]):
#         if ccOnly['image_name'].str.contains(filename).any():
#             src = os.path.join(databases[i],filename)
#             shutil.copy(src, dest)
#             count+=1
#             print(count)

path, dirs, files = next(os.walk(dest))
file_count = len(files)
print("testing github")
print("There are", file_count, "in the CC only directory.")

count = 0
### To make sure all images in the excel file are in the cc only folder
notIn = []
for i in range(len(ccOnly)):
    excelName = ccOnly['image_name'][i]
    for root, dirs, files in os.walk(dest):
        if excelName in files:
            count+=1
        else:
            print("This image isnt in the folder.")
            notIn.append(excelName)

## To check how many malignant (1/2) vs benign cases (3/5)
ccOnly['Mass Type'].value_counts()

###To find which cases have more than one mass
# imageNamesOnly = massCenterMarks['image_name']
# duplicates = imageNamesOnly.duplicated(keep=False)
# dupIndex = duplicates[duplicates].index ###using boolean indexing
# duplicatedNames = imageNamesOnly[dupIndex]
# dupNameCounts = duplicatedNames.value_counts()
# dupNameCounts.to_excel('duplicates.xlsx')
