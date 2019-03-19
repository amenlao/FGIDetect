import  os
if __name__ == '__main__':
    from PyInstaller.__main__ import run
    opts=['ui_setup.py','-w','--icon=ico/gesture.png']
    run(opts)
