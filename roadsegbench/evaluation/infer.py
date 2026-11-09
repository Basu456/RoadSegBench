from pathlib import Path
import numpy as np, pandas as pd, torch, cv2, json
from torch.utils.data import DataLoader
from roadsegbench.data.dataset import RoadManifestDataset
from roadsegbench.models.registry import build_model
from roadsegbench.metrics import binary_metrics, boundary_iou

def evaluate_checkpoint(frame, roots, decoder, architecture, model_cfg, checkpoint,
                        output_dir, size=(512,1024), batch_size=2, threshold=.5,
                        boundary_radius=2, workers=2):
    device="cuda" if torch.cuda.is_available() else "cpu"
    ds=RoadManifestDataset(frame,roots,decoder,size,augment=False)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=workers)
    model=build_model(architecture,**model_cfg).to(device)
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(state["state_dict"]); model.eval()
    out=Path(output_dir); (out/"predictions").mkdir(parents=True,exist_ok=True)
    rows=[]
    with torch.no_grad():
        for x,y,ids in loader:
            logits=model(x.to(device))
            logits=torch.nn.functional.interpolate(logits,size=y.shape[-2:],mode="bilinear",align_corners=False)
            probs=torch.sigmoid(logits).cpu().numpy()[:,0]
            targets=y.numpy()[:,0]
            for image_id,p,t in zip(ids,probs,targets):
                pred=(p>=threshold).astype(np.uint8)
                safe=str(image_id).replace("/","_").replace("\\","_")
                pred_path=out/"predictions"/f"{safe}.png"
                cv2.imwrite(str(pred_path),pred*255)
                m=binary_metrics(p,t,threshold)
                m["biou"]=boundary_iou(p,t,threshold,boundary_radius)
                rows.append({"image_id":image_id,**m,"prediction_path":str(pred_path)})
    result=pd.DataFrame(rows)
    result.to_csv(out/"per_image_metrics.csv",index=False)
    return result
