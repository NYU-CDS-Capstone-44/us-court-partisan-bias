import bz2
import shutil

# Decompressing the file
with bz2.BZ2File("/vast/amr10211/opinions-2023-08-31.csv.bz2", 'rb') as fr, open("/vast/amr10211/opinions-data.csv", "wb") as fw:
    shutil.copyfileobj(fr, fw)

