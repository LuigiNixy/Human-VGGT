# THuman2.0 acquisition and rendering

1. Request THuman2.0 through the official release procedure and review the
   noncommercial research terms. Track the release version, archive hash and
   access restrictions. Use the released clothed scan meshes and textures.
2. Inventory mesh and texture integrity, confirmed scale, and source-to-world
   axis transforms. Verify identity grouping before freezing train, validation,
   and test splits. Use the first ten authorized scans for plumbing and record
   their role before any model comparison.
3. Version a headless Blender job with its Blender version, color management,
   camera azimuth/elevation schedule, exact K and world-to-camera transform,
   and seed for lighting/background draws. A starting schedule uses two
   elevations with eight evenly spaced azimuths each. Keep 1/2/4/8-view tuples
   fixed per sample. Export sRGB PNG and binary foreground mask, optical-axis
   meter depth, and mesh triangle IDs or barycentric surface anchors when available.
4. Verify renderer axes and depth against the calibrated synthetic fixture. If a
   depth pass reports Euclidean ray range, convert at each pixel using
   `z = range / ||K^-1 [x,y,1]||`. Validate on an oblique surface and compare
   projected GT points to the rendered z-buffer in a second view.
5. Store checksums, renderer settings, and rendering failures in a dataset
   manifest. Keep authored scan assets and derivatives in authorized storage.
   Evaluate visible point samples and complete scan coverage with separate
   protocols.

Source [THuman2.0](https://github.com/ytrock/THuman2.0-Dataset)
