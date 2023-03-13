# IANet
Official code repository for paper **Local-Global Interaction and Progressive Aggregation for Video Salient Object Detection** (ICONIP 2022)

## Usage

**Each dataset corresponds to a txt path file, with each row arranged by img_path, gt_path and flow_path.**

## Training
1. Download the train dataset (contains DAVIS16, DAVSOD and DUTS-TR) from [Baidu Driver](https://pan.baidu.com/s/1F2RrcgJylUMYkWiUAoaL2A)(PSW:wuqv).
2. Download the pre_trained ResNet34 [backbone](https://download.pytorch.org/models/resnet34-333f7ec4.pth) to your specified folder.
3. The training of entire model is implemented on a NVIDIA TiTAN X GPUs:
- Run `python main.py --mode=train`

## Testing
1. Download the test dataset  (contains DAVIS16, DAVSOD, FBMS, SegTrack-V2 and VOS) from [Baidu Driver](https://pan.baidu.com/s/1F2RrcgJylUMYkWiUAoaL2A)(PSW:wuqv).
2. Download the final trained model from [Baidu Driver](https://pan.baidu.com/s/1IPwwghNX4GrBOgKWT2NO0w)(PSW:u9wa).
3. Run `python main.py --mode=test`.

## Result
1. The saliency maps can be download from [Baidu Driver](https://pan.baidu.com/s/15KvCtIZ8BQ3zhdkVVf73wg) (PSW: u76y).
2. Evaluation Toolbox: We use the standard evaluation toolbox from [DAVSOD benchmark](https://github.com/DengPingFan/DAVSOD).

## Citation
Please cite the following paper if you use this repository in your research.