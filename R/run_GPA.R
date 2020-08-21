library("geomorph")


gpa <- function (A){
	nlandmarks <- dim(A)[1]
	nshapes    <- dim(A)[3]
	B      <- gpagen(A)  
	r      <- aperm(B$coords, c(1,3,2) )
	r      <- as.vector( r )
	dim(r) <- c(nlandmarks*nshapes, 2)
	r
}

process.data <- function (a){
	SHAPE      <- a[,1]
	LANDMARK   <- a[,2]
	nshapes    <- max(SHAPE)  # number of shapes
	nlandmarks <- max(LANDMARK)  # number of landmarks
	A          <- as.vector( t(a[,3:4]) )
	dim(A)     <- c(2,nlandmarks,nshapes)
	A          <- aperm(A, c(2,1,3) )
	r          <- gpa(A)
	X          <- r[,1]
	Y          <- r[,2]
	df         <- data.frame(SHAPE=SHAPE, LANDMARK=LANDMARK, X=X, Y=Y)
	df$SHAPE   <- formatC(df$SHAPE, format = 'd')
	df$SHAPE   <- formatC(df$LANDMARK, format = 'd')
	df$X       <- formatC(df$X, digits = 3, format = 'f')
	df$Y       <- formatC(df$Y, digits = 3, format = 'f')
	df
}


#(0) Conduct GPA for one dataset:
dirREPO <- dirname( dirname( sys.frame(1)$ofile ) )
dir0    <- file.path(dirREPO, "Data", "Landmarks", "1_Original")
name    <- "Bell"
fname0  <- paste( file.path(dir0, name), ".csv", sep="")
a       <- read.csv(fname0)   # 4 columns: shape, landmark, x, y
df      <- process.data(a)



# #(1) Conduct GPA for all files:
# dirREPO <- dirname( dirname( sys.frame(1)$ofile ) )
# dir0    <- file.path(dirREPO, "Data", "Landmarks", "1_Original")
# dir1    <- file.path(dirREPO, "Data", "Landmarks", "2_GPA")
# names  <- c('Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key')

# for (name in names){
	# fname0 <- paste( file.path(dir0, name), ".csv", sep="")
	# fname1 <- paste( file.path(dir1, name), ".csv", sep="")
	# a      <- read.csv(fname0)   # 4 columns: shape, landmark, x, y
	# df     <- process.data(a)
	# write.table(df, file=fname1, row.names=F, sep=",")
# }





