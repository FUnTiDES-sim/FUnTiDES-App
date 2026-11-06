# DAS validation

- Untar the archive `DAS_validation_data.tar` containing all the data

- Execute the script `script_Chengyi/dataview_FX2TX_3D.py` to visualize and save the figure representing all the results

- This code treats the hawen data in frequency to get the DAS results while the other results (analytical, GEOSX and gar6more) are read directly

- The other scripts in the folder `script_Chengyi` are pre- or post-treatment codes:
    - `acqui_DAS.py` : acquisition and DAS projection
    - `class_FX.py` : IFFT and FFT
    - `phy_SOL.py` : calculation of the stiffness C