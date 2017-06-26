
#
# Results of...
# $ grep "require external data" ./travis_test_lists/testlist_minimal_sorted.txt | grep -o "(iris[^ ]*" | grep -o "[^()]*" | sort | uniq
#

selected_testdata_test_names = [
    'iris.tests.experimental.regrid.test_regrid_area_weighted_rectilinear_src_and_grid.TestAreaWeightedRegrid',
    'iris.tests.experimental.test_fieldsfile.TestStructuredLoadFF',
    'iris.tests.experimental.test_fieldsfile.TestStructuredLoadPP',
    'iris.tests.integration.experimental.test_regrid_ProjectedUnstructured.TestProjectedUnstructured',
    'iris.tests.integration.plot.test_nzdateline.TestExtent',
    'iris.tests.integration.test_aggregated_cube.Test_aggregated_by',
    'iris.tests.integration.test_ff.TestFFGrid',
    'iris.tests.integration.test_ff.TestLBC',
    'iris.tests.integration.test_ff.TestSkipField',
    'iris.tests.integration.test_FieldsFileVariant.TestCreate',
    'iris.tests.integration.test_FieldsFileVariant.TestRead',
    'iris.tests.integration.test_FieldsFileVariant.TestUpdate',
    'iris.tests.integration.test_grib2.TestAsCubes',
    'iris.tests.integration.test_grib2.TestDRT3',
    'iris.tests.integration.test_grib2.TestGDT30',
    'iris.tests.integration.test_grib2.TestGDT40',
    'iris.tests.integration.test_grib2.TestGDT5',
    'iris.tests.integration.test_grib2.TestImport',
    'iris.tests.integration.test_grib2.TestPDT11',
    'iris.tests.integration.test_grib2.TestPDT8',
    'iris.tests.integration.test_grib_load.TestBasicLoad',
    'iris.tests.integration.test_grib_load.TestIjDirections',
    'iris.tests.integration.test_grib_load.TestShapeOfEarth',
    'iris.tests.integration.test_netcdf.TestCellMeasures',
    'iris.tests.integration.test_netcdf.TestCoordSystem',
    'iris.tests.integration.test_netcdf.TestHybridPressure',
    'iris.tests.integration.test_netcdf.TestLazySave',
    'iris.tests.integration.test_netcdf.TestSaveMultipleAuxFactories',
    'iris.tests.integration.test_new_axis.Test',
    'iris.tests.integration.test_PartialDateTime.Test',
    'iris.tests.integration.test_pickle.test_ff',
    'iris.tests.integration.test_pickle.TestGribMessage',
    'iris.tests.integration.test_pickle.test_netcdf',
    'iris.tests.integration.test_pickle.test_pp',
    'iris.tests.integration.test_pp_constrained_load_cubes.Test',
    'iris.tests.integration.test_pp.TestAsCubes',
    'iris.tests.integration.test_pp.TestCallbackLoad',
    'iris.tests.integration.test_pp.TestLoadLittleendian',
    'iris.tests.integration.test_pp.TestLoadPartialMask',
    'iris.tests.integration.test_pp.TestZonalMeanBounds',
    'iris.tests.integration.test_regridding.TestGlobalSubsample',
    'iris.tests.integration.test_regridding.TestOSGBToLatLon',
    'iris.tests.integration.test_regridding.TestUnstructured',
    'iris.tests.integration.test_trajectory.TestColpex',
    'iris.tests.integration.test_trajectory.TestTriPolar',
    'iris.tests.test_abf.TestAbfLoad',
    'iris.tests.test_analysis_calculus.TestCubeDelta',
    'iris.tests.test_analysis.TestAnalysisBasic',
    'iris.tests.test_analysis.TestAnalysisWeights',
    'iris.tests.test_analysis.TestAreaWeightGeneration',
    'iris.tests.test_analysis.TestAreaWeights',
    'iris.tests.test_analysis.TestLatitudeWeightGeneration',
    'iris.tests.test_analysis.TestRotatedPole',
    'iris.tests.test_basic_maths.TestBasicMaths',
    'iris.tests.test_basic_maths.TestDivideAndMultiply',
    'iris.tests.test_basic_maths.TestExponentiate',
    'iris.tests.test_basic_maths.TestLog',
    'iris.tests.test_cdm.TestConversionToCoordList',
    'iris.tests.test_cdm.TestCubeCollapsed',
    'iris.tests.test_cdm.TestCubeExtract',
    'iris.tests.test_cdm.TestCubeStringRepresentations',
    'iris.tests.test_cdm.TestDataManagerIndexing',
    'iris.tests.test_cdm.TestMaskedData',
    'iris.tests.test_cdm.TestStockCubeStringRepresentations',
    'iris.tests.test_cdm.TestTrimAttributes',
    'iris.tests.test_cdm.TestValidity',
    'iris.tests.test_cf.TestCFReader',
    'iris.tests.test_cf.TestClimatology',
    'iris.tests.test_cf.TestLabels',
    'iris.tests.test_cf.TestLoad',
    'iris.tests.test_constraints.TestConstraints',
    'iris.tests.test_constraints.TestCubeExtract',
    'iris.tests.test_constraints.TestCubeListConstraint',
    'iris.tests.test_constraints.TestCubeListStrictConstraint',
    'iris.tests.test_constraints.TestCubeLoadConstraint',
    'iris.tests.test_constraints.TestSimple',
    'iris.tests.test_coord_api.TestCoordIntersection',
    'iris.tests.test_coord_api.TestCoord_ReprStr_nontime',
    'iris.tests.test_coord_api.TestCoord_ReprStr_time',
    'iris.tests.test_coord_api.TestCoordSlicing',
    'iris.tests.test_coord_api.TestGetterSetter',
    'iris.tests.test_cube_to_pp.TestPPSave',
    'iris.tests.test_cube_to_pp.TestPPSaveRules',
    'iris.tests.test_ff.TestFF2PP2Cube',
    'iris.tests.test_ff.TestFFHeader',
    'iris.tests.test_ff.TestFFieee32',
    'iris.tests.test_ff.TestFFVariableResolutionGrid',
    'iris.tests.test_file_load.TestFileLoad',
    'iris.tests.test_file_save.TestSaveDot',
    'iris.tests.test_file_save.TestSaveInvalid',
    'iris.tests.test_file_save.TestSavePP',
    'iris.tests.test_file_save.TestSaver',
    'iris.tests.test_grib_save.TestCubeSave',
    'iris.tests.test_grib_save.TestLoadSave',
    'iris.tests.test_hybrid.TestHybridPressure',
    'iris.tests.test_hybrid.TestRealistic4d',
    'iris.tests.test_image_json.TestImageFile',
    'iris.tests.test_interpolation.TestNearestLinearInterpolRealData',
    'iris.tests.test_interpolation.TestNearestNeighbour',
    'iris.tests.test_interpolation.TestNearestNeighbour__Equivalent',
    'iris.tests.test_io_init.TestFileFormatPicker',
    'iris.tests.test_iterate.TestIterateFunctions',
    'iris.tests.test_load.TestLoad',
    'iris.tests.test_load.TestLoadCube',
    'iris.tests.test_load.TestLoadCubes',
    'iris.tests.test_mapping.TestBasic',
    'iris.tests.test_mapping.TestBoundedCube',
    'iris.tests.test_mapping.TestLimitedAreaCube',
    'iris.tests.test_mapping.TestLowLevel',
    'iris.tests.test_mapping.TestMappingSubRegion',
    'iris.tests.test_mapping.TestUnmappable',
    'iris.tests.test_merge.TestColpex',
    'iris.tests.test_merge.TestDataMerge',
    'iris.tests.test_merge.TestMultiCube',
    'iris.tests.test_merge.TestSingleCube',
    'iris.tests.test_name.TestLoad',
    'iris.tests.test_netcdf.TestNetCDFLoad',
    'iris.tests.test_netcdf.TestNetCDFSave',
    'iris.tests.test_netcdf.TestNetCDFUKmoProcessFlags',
    'iris.tests.test_netcdf.TestSave',
    'iris.tests.test_nimrod.TestLoad',
    'iris.tests.test_pickling.TestPickle',
    'iris.tests.test_plot.Test1dPlotMultiArgs',
    'iris.tests.test_plot.Test1dQuickplotPlotMultiArgs',
    'iris.tests.test_plot.Test1dQuickplotScatter',
    'iris.tests.test_plot.Test1dScatter',
    'iris.tests.test_plot.TestAttributePositive',
    'iris.tests.test_plot.TestContour',
    'iris.tests.test_plot.TestContourf',
    'iris.tests.test_plot.TestHybridHeight',
    'iris.tests.test_plot.TestMissingCoord',
    'iris.tests.test_plot.TestMissingCS',
    'iris.tests.test_plot.TestPcolor',
    'iris.tests.test_plot.TestPcolormesh',
    'iris.tests.test_plot.TestPcolormeshNoBounds',
    'iris.tests.test_plot.TestPcolorNoBounds',
    'iris.tests.test_plot.TestPlot',
    'iris.tests.test_plot.TestPlotCoordinatesGiven',
    'iris.tests.test_plot.TestPlotDimAndAuxCoordsKwarg',
    'iris.tests.test_plot.TestPlotOtherCoordSystems',
    'iris.tests.test_plot.TestQuickplotPlot',
    'iris.tests.test_plot.TestSimple',
    'iris.tests.test_pp_cf.TestAll',
    'iris.tests.test_pp_module.TestPackedPP',
    'iris.tests.test_pp_module.TestPPCopy',
    'iris.tests.test_pp_module.TestPPField_GlobalTemperature',
    'iris.tests.test_pp_module.TestPPFile',
    'iris.tests.test_pp_module.TestPPFileExtraXData',
    'iris.tests.test_pp_module.TestPPFileWithExtraCharacterData',
    'iris.tests.test_pp_stash.TestPPStash',
    'iris.tests.test_pp_to_cube.TestPPLoadCustom',
    'iris.tests.test_pp_to_cube.TestPPLoading',
    'iris.tests.test_pp_to_cube.TestPPLoadRules',
    'iris.tests.test_quickplot.TestLabels',
    'iris.tests.test_quickplot.TestQuickplotCoordinatesGiven',
    'iris.tests.test_quickplot.TestTimeReferenceUnitsLabels',
    'iris.tests.test_regrid.TestRegrid',
    'iris.tests.test_uri_callback.TestCallbacks',
    'iris.tests.test_util.TestAsCompatibleShape',
    'iris.tests.test_util.TestDescribeDiff',
    'iris.tests.test_verbose_fileformat_rules_logging.TestVerboseLogging',
    'iris.tests.unit.analysis.cartography.test_project.TestAll',
    'iris.tests.unit.analysis.cartography.test__xy_range.Test',
    'iris.tests.unit.analysis.geometry.test_geometry_area_weights.Test',
    'iris.tests.unit.analysis.interpolate.test_linear.Test_masks',
    'iris.tests.unit.analysis.interpolation.test_get_xy_dim_coords.TestGetXYCoords',
    'iris.tests.unit.analysis.maths.test_add.TestBroadcasting',
    'iris.tests.unit.analysis.maths.test_divide.TestBroadcasting',
    'iris.tests.unit.analysis.maths.test_multiply.TestBroadcasting',
    'iris.tests.unit.analysis.maths.test_subtract.TestBroadcasting',
    'iris.tests.unit.analysis.regrid.test_RectilinearRegridder.Test___call____circular',
    'iris.tests.unit.analysis.regrid.test_RectilinearRegridder.Test___call____NOP',
    'iris.tests.unit.analysis.regrid.test_RectilinearRegridder.Test___call____rotated_to_lat_lon',
    'iris.tests.unit.analysis.stats.test_pearsonr.Test',
    'iris.tests.unit.analysis.trajectory.test_interpolate.TestFailCases',
    'iris.tests.unit.aux_factory.test_AuxCoordFactory.Test_lazy_aux_coords',
    'iris.tests.unit.cube.test_Cube.Test_slices_over',
    'iris.tests.unit.experimental.equalise_cubes.test_equalise_attributes.TestEqualiseAttributes',
    'iris.tests.unit.experimental.regrid.test__CurvilinearRegridder.Test___call__',
    'iris.tests.unit.experimental.regrid.test__CurvilinearRegridder.Test___call____bad_src',
    'iris.tests.unit.experimental.um.test_FieldsFileVariant.Test_class_assignment',
    'iris.tests.unit.experimental.um.test_FieldsFileVariant.Test_filename',
    'iris.tests.unit.experimental.um.test_FieldsFileVariant.Test_mode',
    'iris.tests.unit.fileformats.ff.test_FFHeader.Test_integer_constants',
    'iris.tests.unit.fileformats.test_rules.TestConcreteReferenceTarget',
    'iris.tests.unit.fileformats.test_rules.TestLoadCubes',
    'iris.tests.unit.util.test_demote_dim_coord_to_aux_coord.Test',
    'iris.tests.unit.util.test_promote_aux_coord_to_dim_coord.Test',
    'iris.tests.unit.util.test_unify_time_units.Test',
]