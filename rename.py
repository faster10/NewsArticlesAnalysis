import glob, os

def rename(dir, pattern, titlePattern):
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        os.rename(pathAndFilename, 
                  os.path.join(dir, titlePattern % title + ext))

rename(r'SatireArticlesTest/', r'*.txt', r'sat_(%s)')
rename(r'SeriousArticlesTest/', r'*.txt', r'ser_(%s)')