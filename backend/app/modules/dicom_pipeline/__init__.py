"""
dicom_pipeline — DICOM MRA to IntrA-compatible mesh preprocessing.

Pipeline stages:
    loader      → decode a zipped DICOM series into a 3D volume
    segmenter   → threshold/extract vessel mask from the volume
    mesher      → convert binary mask to a surface mesh
    cropper     → crop mesh around a user-clicked aneurysm location
    harmonizer  → resample/normalize to match IntrA training distribution
    session_store → in-memory session state for multi-step endpoints
"""
