import os
import shutil
from glob import glob
from matplotlib import matplotlib_fname
from matplotlib import get_cachedir
        
def create_mpl_theme():
    """
    Shamelessly adapted from
    https://stackoverflow.com/questions/40290004/how-can-i-configure-matplotlib-to-be-able-to-read-fonts-from-a-local-path
    """    

    #Load environment
    conda_env_dir = os.getenv('CONDA_PREFIX')
    print('Currently on conda environment:', conda_env_dir)
 
    #Get current file folder and subfolders for fonts and styles
    dir_src = os.path.dirname(__file__)
    dir_src_fonts  = os.path.join(dir_src, 'fonts')
    dir_src_styles = os.path.join(dir_src, 'styles')

    dir_dest = os.path.dirname(matplotlib_fname())
    dir_dest_fonts  = os.path.join(dir_dest, 'fonts', 'ttf')
    dir_dest_styles = os.path.join(dir_dest, 'stylelib')

    #Install styles and fonts
    answer = input("Install fonts and styles? [y/N] ")
    if answer.lower() in ["y","yes"]:
        
        #Copy fonts
        print(f'Transfering .ttf and .otf files from {dir_src_fonts} to {dir_dest_fonts}.')
        for file in glob(os.path.join(dir_src_fonts, '*.[ot]tf')):
            if not os.path.exists(os.path.join(dir_dest_fonts, os.path.basename(file))):
                print(f'Adding font "{os.path.basename(file)}".')
                shutil.copy(file, dir_dest_fonts)

        # Delete font cache
        dir_cache = get_cachedir()
        print(dir_cache)
        for file in glob(os.path.join(dir_cache, '*.cache')) + glob(os.path.join(dir_cache, 'font*')):
            print(file)
            if not os.path.isdir(file): # don't dump the tex.cache folder... because dunno why
                os.remove(file)
                print(f'Deleted font cache {file}.')

        #Copy Styles
        print(f'Transfering .mplstyle files from {dir_src_styles} to {dir_dest_styles}.')
        for file in glob(os.path.join(dir_src_styles, '*.mplstyle')):
            if not os.path.exists(os.path.join(dir_dest_styles, os.path.basename(file))):
                print(f'Adding style "{os.path.basename(file)}".')
                shutil.copy(file, dir_dest_styles)

        print("Installation complete!")
        print("Please restart python for the changes to take place.")
    else:
        print('No fonts or styles installed')
    