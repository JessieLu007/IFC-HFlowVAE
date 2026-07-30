# set_fonts.py
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties, findfont

def apply_times_with_cn(times_name="Times New Roman", cn_font_path=None, cn_font_name_candidates=None):
    
    mpl.rcParams['axes.unicode_minus'] = False

    # 1) 如果提供了中文字体文件路径，注册并使用它
    if cn_font_path:
        try:
            font_manager.fontManager.addfont(cn_font_path)
            # 有些 matplotlib 需要重建缓存（可选）
            try:
                font_manager._rebuild()
            except Exception:
                pass
            
            cn_fp = FontProperties(fname=cn_font_path)
            cn_name = cn_fp.get_name()
        except Exception:
            cn_name = None
    else:
        cn_name = None


    if not cn_name:
        cn_name = None
        if cn_font_name_candidates is None:
            cn_font_name_candidates = [
                "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
                "PingFang", "Hiragino Sans GB", "Arial Unicode MS"
            ]
        for f in font_manager.fontManager.ttflist:
            for cand in cn_font_name_candidates:
                if cand.lower() in f.name.lower():
                    cn_name = f.name
                    break
            if cn_name:
                break

    
    family_list = [times_name]
    if cn_name:
        family_list.append(cn_name)
    family_list += ["DejaVu Sans"] 
    mpl.rcParams['font.family'] = family_list
    mpl.rcParams['font.serif'] = [times_name, "Times", "DejaVu Serif"]

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    en_fp = FontProperties(family=times_name)
    cn_fp = FontProperties(fname=cn_font_path) if cn_font_path else (FontProperties(family=cn_name) if cn_name else None)

    print("Applied font.family:", mpl.rcParams['font.family'])
    try:
        print("findfont Times ->", findfont(en_fp))
        if cn_fp:
            print("findfont CN ->", findfont(cn_fp))
    except Exception:
        pass

    return en_fp, cn_fp
