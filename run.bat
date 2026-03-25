python -m src.port_pipeline.scan_cli --place "singapore" --size 1280x1280 --meters 2000 --detector-format json --mmrotate-config mmrotate/oriented_rcnn_r50_fpn_1x_dota_le90.py --mmrotate-checkpoint mmrotate/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --output-dir outputs_scan_5000 --verbose --max-no-new 3 --step-sleep-seconds 0.2


python -m src.port_pipeline.scan_cli --place "singapore" --size 960x960 --meters 2000 --detector-format json --mmrotate-config mmrotate/oriented_rcnn_r50_fpn_1x_dota_le90.py --mmrotate-checkpoint mmrotate/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --output-dir outputs_scan_2000 --verbose --max-no-new 5 --step-sleep-seconds 0.2


python -m src.port_pipeline.scan_cli --place "singapore" --size 960x960 --meters 2000 --detector-format json --mmrotate-config mmrotate/oriented_rcnn_r50_fpn_1x_dota_le90.py --mmrotate-checkpoint mmrotate/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --output-dir outputs_scan_2000 --verbose --max-no-new 5 --step-sleep-seconds 0.6
