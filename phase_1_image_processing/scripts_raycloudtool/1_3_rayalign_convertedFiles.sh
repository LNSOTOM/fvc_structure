cd /media/laura/laura_usb/code/fvc_structure/raycloudtools/build

# Align L2 (moving) onto L1 (fixed):
./bin/rayalign-0.0.1 \
  "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar_l2/cloud2d765c57b8471e2c.ply" \
  "/media/laura/laura_usb/qgis/calperumResearch/site1_1_DD0001/lidar/20220519_SASMDD0001_triple_AusGeoid.ply" \
  --verbose | tee \
/media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/output/rayimport_samples/align_L2_to_L1_raw.log
