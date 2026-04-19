"""
Vessel segmentation for TOF-MRA volumes.

TOF-MRA enhances flowing blood, so vessels appear as the brightest voxels
on a mostly dark background. A percentile threshold followed by a morphological
cleanup and largest-connected-component selection is enough to extract the
main vascular tree as a single mask — we do NOT try to label individual
vessels or anatomical regions.
"""

from __future__ import annotations

import logging

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


# Default threshold 99.0 was tuned against Lausanne TOF-MRA data:
# 95.0 produced a vessel+tissue blob, 98.0 still fused vessels,
# 99.0 produces clean tubular structures matching IntrA's
# single-segment convention (~10-20k voxels, ~12k vertices).
def segment_vessels(
    volume: sitk.Image,
    threshold_percentile: float = 99.0,
    use_vesselness: bool = False,
) -> sitk.Image:
    """
    Extract the vascular tree from an MRA volume.

    Steps:
        1. (optional) Hessian vesselness enhancement.
        2. Intensity threshold at ``threshold_percentile`` of non-zero voxels.
        3. Morphological opening with radius-1 ball to kill speckle.
        4. Keep the single largest connected component.

    Args:
        volume: 3D TOF-MRA volume.
        threshold_percentile: Percentile (over non-zero voxels) used as
            the intensity cutoff. Defaults to 99 (top 1%).
        use_vesselness: If True, run the Hessian vesselness filter first
            and threshold the vesselness response instead of raw intensity.

    Returns:
        Binary ``sitk.Image`` (uint8, values {0, 1}) with the input's
        spacing, origin, and direction preserved.
    """
    source = refine_with_vesselness(volume) if use_vesselness else volume
    source_f = sitk.Cast(source, sitk.sitkFloat32)

    arr = sitk.GetArrayViewFromImage(source_f)
    nonzero = arr[arr > 0]
    if nonzero.size == 0:
        raise ValueError("Volume has no non-zero voxels; nothing to segment.")
    threshold = float(np.percentile(nonzero, threshold_percentile))
    logger.info(
        "Segmentation threshold: p%.1f = %.4f (over %d non-zero voxels)",
        threshold_percentile, threshold, int(nonzero.size),
    )

    mask = sitk.BinaryThreshold(
        source_f,
        lowerThreshold=threshold,
        upperThreshold=float(arr.max()) + 1.0,
        insideValue=1,
        outsideValue=0,
    )

    mask = sitk.BinaryMorphologicalOpening(mask, [1, 1, 1])

    # Largest connected component.
    components = sitk.ConnectedComponent(mask)
    relabeled = sitk.RelabelComponent(components, sortByObjectSize=True)
    largest = sitk.BinaryThreshold(
        relabeled, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0,
    )
    largest = sitk.Cast(largest, sitk.sitkUInt8)
    largest.CopyInformation(volume)
    return largest


def refine_with_vesselness(volume: sitk.Image) -> sitk.Image:
    """
    Hessian-based vesselness enhancement (Frangi-style).

    Uses SimpleITK's ObjectnessMeasureImageFilter configured for
    tubular structures. Returns a float image where bright values
    mean "looks like a vessel"; it is not itself a binary mask.
    """
    float_vol = sitk.Cast(volume, sitk.sitkFloat32)
    hessian = sitk.HessianRecursiveGaussianImageFilter()
    hessian.SetSigma(1.0)
    h = hessian.Execute(float_vol)

    objectness = sitk.ObjectnessMeasureImageFilter()
    objectness.SetBrightObject(True)
    objectness.SetScaleObjectnessMeasure(True)
    objectness.SetObjectDimension(1)  # 1 = tubular (vessels)
    return objectness.Execute(h)
