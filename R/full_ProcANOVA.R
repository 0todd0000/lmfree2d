library("geomorph")


process.data <- function (a){
	SHAPE      <- a[,1]
	nshapes    <- max(SHAPE)  # number of shapes
	npoints    <- length(SHAPE) / nshapes  # number of landmarks
	A          <- as.vector( t(a[,2:3]) )
	dim(A)     <- c(2,npoints,nshapes)
	A          <- aperm(A, c(2,1,3) )
	B          <- gpagen(A) 
	group      <- factor( c(0,0,0,0,0, 1,1,1,1,1) )   # group labels
	gdf        <- geomorph.data.frame(B, group=group) # geomorph data frame
	fit        <- procD.lm(coords ~ group, data = gdf, iter = 1000, RRPP = FALSE, print.progress = FALSE) 
	s          <- summary(fit)
	F          <- s$table$F[1]
	p          <- s$table$"Pr(>F)"[1]
	c(F, p)
}


# #(0) Conduct Procrustes ANOVA for one dataset:
# dirREPO <- dirname( dirname( sys.frame(1)$ofile ) )
# name    <- "Bell"
# fname0  <- file.path(dirREPO, "Data", name, "geom_sroc.csv")
# a       <- read.csv(fname0)   # 3 columns: shape, x, y
# res     <- process.data(a)  # F and p values
# print(res)






#(1) Conduct Procrustes ANOVA for all datasets:
dirREPO <- dirname( dirname( sys.frame(1)$ofile ) )
names   <- c('Bell', 'Comma', 'Device8', 'Face', 'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key')
F       <- numeric(9)
p       <- numeric(9)
for (i in 1:9){
	fname0 <- file.path(dirREPO, "Data", names[i], "geom_sroc.csv")
	a      <- read.csv(fname0)   # 4 columns: shape, landmark, x, y
	res    <- process.data(a)
	F[i]   <- res[1]
	p[i]   <- res[2]
}
# assemble results into a data frame:
df      <- data.frame(Name=names, F=F, p=p)
df$F    <- formatC(df$F, digits = 3, format = 'f')
df$p    <- formatC(df$p, digits = 3, format = 'f')
# save:
fname1  <- file.path(dirREPO, "Results", "full_ProcANOVA.csv")
write.table(df, file=fname1, row.names=FALSE , sep=",", quote = FALSE)



