"""Extract approximate read-offs from the printed Figure 4.11 vector curve.
Usage: python read_plot.py /path/to/bv_cvxbook.pdf
These are drawing-coordinate estimates, not solutions for unavailable A,b.
"""
from pathlib import Path
import sys,json
import fitz
page=fitz.open(sys.argv[1])[198]
draw=next(d for d in page.get_drawings() if d['type']=='s' and d['width'] and .8<d['width']<.95 and len(d['items'])==42)
raw=[draw['items'][0][1]]+[s[2] for s in draw['items']]
x0,x10,y0,y10=191.1020050048828,322.8260192871094,283.9339904785156,179.9949951171875
points=[((p.x-x0)*10/(x10-x0),(y0-p.y)*10/(y0-y10)) for p in raw]
intersections=[]
for (x1,y1),(x2,y2) in zip(points,points[1:]):
 if (y1-1)*(y2-1)<=0 and y1!=y2:intersections.append(x1+(1-y1)*(x2-x1)/(y2-y1))
best=min(points,key=lambda p:sum(p))
result={'source_pdf_page':199,'figure':'4.11','interpretation':'Both plotted axes are squared norms. Values are approximate drawing read-offs.','least_squares_point':points[0],'origin_point':points[-1],'F1_at_F2_1':intersections,'minimum_F1_plus_F2_on_polyline':sum(best),'support_point':best}
out=Path(__file__).with_name('read_plot.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
