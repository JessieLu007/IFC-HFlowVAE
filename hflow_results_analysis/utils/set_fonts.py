# set_fonts.py
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties, findfont

def apply_times_with_cn(times_name="Times New Roman", cn_font_path=None, cn_font_name_candidates=None):
    """
    统一设置：英文优先 Times New Roman，中文使用系统可用的中文字体或指定的字体文件路径。
    - times_name: 英文字体名（通常 'Times New Roman'）
    - cn_font_path: 若提供，优先使用该中文字体文件路径（.ttf/.ttc）
    - cn_font_name_candidates: 若未提供路径，按候选名字在系统字体中查找（列表）
    """
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
            # 直接使用文件名对应的字体名会更稳，但这里把路径注册后使用名字回退也行
            cn_fp = FontProperties(fname=cn_font_path)
            cn_name = cn_fp.get_name()
        except Exception:
            cn_name = None
    else:
        cn_name = None

    # 2) 若没有 path，尝试在系统字体中按候选名查找第一个可用项
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

    # 3) 设置 rcParams：让 matplotlib 逐字符尝试列表内字体（英文先 Times，再中文回退）
    family_list = [times_name]
    if cn_name:
        family_list.append(cn_name)
    family_list += ["DejaVu Sans"]  # 保险回退
    mpl.rcParams['font.family'] = family_list
    mpl.rcParams['font.serif'] = [times_name, "Times", "DejaVu Serif"]
    # 保存/导出时嵌入 TrueType（推荐）
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    # 返回用于单项强制的 FontProperties（若需要）
    en_fp = FontProperties(family=times_name)
    cn_fp = FontProperties(fname=cn_font_path) if cn_font_path else (FontProperties(family=cn_name) if cn_name else None)

    # 打印检查信息（可注释）
    print("Applied font.family:", mpl.rcParams['font.family'])
    try:
        print("findfont Times ->", findfont(en_fp))
        if cn_fp:
            print("findfont CN ->", findfont(cn_fp))
    except Exception:
        pass

    return en_fp, cn_fp
