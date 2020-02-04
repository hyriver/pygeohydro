#!/usr/bin/env Rscript

nhdplus <- function(coords=NULL,
                    nwis_id=NULL,
                    out_dir='data',
                    gis_dir='gis_data'){
    library(nhdplusTools)
    library(sf)

    if (!is.null(coords) && is.null(nwis_id)) {
        start_point <- sf::st_sfc(sf::st_point(coords), crs = 4269)
        start_comid <- discover_nhdplus_id(start_point)
        nldi_feature <- list(featureSource = "comid",
                             featureID = start_comid)
    } else if (is.null(coords) && !is.null(nwis_id)) {
        nldi_feature <- list(featureSource = "nwissite",
                             featureID = paste0("USGS-",nwis_id))
        start_comid <- discover_nhdplus_id(nldi_feature = nldi_feature)
        nwissite <- navigate_nldi(nldi_feature = nldi_feature,
                                  mode = "UT",
                                  data_source = "nwissite")
        start_point <- nwissite[-1]$geometry[1]
    } else {
        stop('Either coords or nwis ID should be specified.')
    }

    subset_file <- tempfile(fileext = ".gpkg")

    flowline_nldi <- navigate_nldi(nldi_feature,
                                   mode = "upstreamMain",
                                   data_source = "")
    df <-subset_nhdplus(comids = flowline_nldi$nhdplus_comid,
                                          output_file = subset_file,
                                          nhdplus_data = "download",
                                          return_data = TRUE,
                                          overwrite = TRUE)

    flowline_main <- df$NHDFlowline_Network
    length <- sum(flowline_main$slopelenkm)*1e3
    slope <- mean(subset(flowline_main, slope > 0)$slope)

    flowline_nldi <- navigate_nldi(nldi_feature,
                                   mode = "upstreamTributaries",
                                   data_source = "")
    df <-subset_nhdplus(comids = flowline_nldi$nhdplus_comid,
                                          output_file = subset_file,
                                          nhdplus_data = "download",
                                          return_data = TRUE,
                                          overwrite = TRUE)
    flowline <- df$NHDFlowline_Network
    catchment <- df$CatchmentSP
    waterbody <- df$NHDWaterbody

    area <- sum(df$NHDFlowline_Network$areasqkm)*1e6
    geometry <- sf::st_union(sf::st_buffer(catchment$geometry, 0))

    data <- paste(c('area', 'length', 'slope'),
                  c(round(area, 0), round(length, 0), round(slope, 6)))

    output_dir <- file.path(out_dir, start_comid)
    gis_dir <- file.path(gis_dir, start_comid)

    if (!dir.exists(output_dir)) {dir.create(output_dir, recursive=TRUE)}
    if (!dir.exists(gis_dir)) {dir.create(gis_dir, recursive=TRUE)}

    f <- file.path(output_dir, 'params.txt')
    writeLines(data, f)

    f <- file.path(gis_dir, 'geometry.shp')
    sf::st_write(geometry, f, delete_dsn=TRUE)

    f <- file.path(output_dir, 'watershed.png')
    png(f, units="in", width=5, height=5, res=300)

    title <- flowline$gnis_name[flowline$gnis_name != ' ']
    title <- title[length(title)]
    plot(sf::st_geometry(catchment), lwd = 0.5, main = title)
    plot(sf::st_geometry(flowline), lwd = 3, col = "blue", add = TRUE)
    plot(sf::st_geometry(flowline_main), lwd = 4, col = "red", add = TRUE)
    plot(start_point, cex = 1.5, lwd = 5, col = "green", add = TRUE)
    legend('topright',
           legend=c("Tributaries", "Main Channel", "USGS Station"),
           col=c("blue", "red", "green"),
           bty = "n",
           lty = 1,
           xpd = TRUE,
           inset = c(0, 1))
    dev.off()
}

main <- function() {
    library("optparse")

    option_list <- list(make_option(c("-i", "--station_id"),
                                   type="character",
                                   default=NULL,
                                   help="USGS station ID",
                                   metavar="character"),
                       make_option(c("-c", "--coords"),
                                   type="double",
                                   default=NULL,
                                   help="output file name",
                                   metavar="double double"),
                       make_option(c("-d", "--data_directory"),
                                   type="character",
                                   default="data",
                                   help=paste("path to data directory",
                                              "[default= %default]"),
                                   metavar="character"),
                       make_option(c("-g", "--gis_directory"),
                                   type="character",
                                   default="gis_data",
                                   help=paste("path to GIS directory",
                                              "[default= %default]"),
                                   metavar="character"));

    opt_parser <- OptionParser(option_list=option_list);
    arguments <- parse_args(opt_parser, positional_arguments = TRUE);
    opt <- arguments$options
    args <- arguments$args

    if (!is.null(opt$coords) && is.null(opt$station_id)){
        nhdplus(coords = as.numeric(c(opt$coords, args)),
                out_dir = opt$data_directory,
                gis_dir = opt$gis_directory)
    } else if (is.null(opt$coords) && !is.null(opt$station_id)){
        nhdplus(nwis_id = opt$station_id,
                out_dir = opt$data_directory,
                gis_dir = opt$gis_directory)
    } else {
        print_help(opt_parser)
        stop("Coordinate or station ID should be provided.", call = FALSE)
    }
}

main()
