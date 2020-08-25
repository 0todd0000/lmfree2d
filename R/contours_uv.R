################################################################################
## Analyze contour data using a univariate (Procrustes ANOVA) approach
################################################################################



library("geomorph")


process.data <- function (a){
	SHAPE      <- a[,1]
	nshapes    <- max(SHAPE)  # number of shapes
	npoints    <- length(SHAPE) / nshapes  # number of landmarks
	group      <- factor( c(0,0,0,0,0, 1,1,1,1,1) )   # group labels
	
	# reshape array into the format expected for GPA
	A          <- as.vector( t(a[,2:3]) )
	dim(A)     <- c(2,npoints,nshapes)
	A          <- aperm(A, c(2,1,3) )
	
	# conduct GPA
	t0         <- Sys.time()
	B          <- gpagen(A) 
	t_gpa      <- Sys.time() - t0  # GPA execution duration
	
	# conduct Procrustes ANOVA
	t0         <- Sys.time()
	gdf        <- geomorph.data.frame(B, group=group) # geomorph data frame
	fit        <- procD.lm(coords ~ group, data = gdf, iter = 1000, RRPP = FALSE, print.progress = FALSE) 
	s          <- summary(fit)
	t_pANOVA   <- Sys.time() - t0
	
	# assemble outputs
	F          <- s$table$F[1]
	p          <- s$table$"Pr(>F)"[1]
	c(F, p, t_gpa, t_pANOVA)
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
t_gpa   <- numeric(9)
t_ANOVA <- numeric(9)
for (i in 1:9){
	fname0     <- file.path(dirREPO, "Data", names[i], "geom_sroc.csv")
	a          <- read.csv(fname0)   # 4 columns: shape, landmark, x, y
	res        <- process.data(a)
	F[i]       <- res[1]
	p[i]       <- res[2]
	t_gpa[i]   <- res[3]
	t_ANOVA[i] <- res[4]
}

# save results:
df          <- data.frame(Name=names, F=F, p=p, t_gpa=t_gpa, t_ANOVA=t_ANOVA)
df$F        <- formatC(df$F,        digits = 3, format = 'f')
df$p        <- formatC(df$p,        digits = 3, format = 'f')
df$t_gpa    <- formatC(df$t_gpa,    digits = 6, format = 'f')
df$t_ANOVA  <- formatC(df$t_ANOVA,  digits = 6, format = 'f')
fname1      <- file.path(dirREPO, "Results", "contours_uv.csv")
write.table(df, file=fname1, row.names=FALSE , sep=",", quote = FALSE)



