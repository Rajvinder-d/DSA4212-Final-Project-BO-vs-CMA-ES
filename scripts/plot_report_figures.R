# Report-only plots from curated results. No optimisation or model evaluation.
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) normalizePath(args[1]) else normalizePath(".")
out <- file.path(root, "report", "figures")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
# Native Quartz avoids an optional X11/Cairo dependency on macOS. Stage each
# file so a graphics-device warning cannot masquerade as a successful rebuild.
save_figure <- function(n, draw, width) {
  mac <- identical(unname(Sys.info()["sysname"]), "Darwin")
  for (format in c("png", "pdf")) {
    target <- file.path(out, paste0("figure", n, ".", format))
    staged <- tempfile(tmpdir=out, fileext=paste0(".", format))
    if (format == "png") {
      png(staged, width=width*300, height=1150, res=300, type=if(mac) "quartz" else "cairo")
    } else if (mac) {
      quartz(type="pdf", file=staged, width=width, height=1150/300)
    } else {
      cairo_pdf(staged, width=width, height=1150/300)
    }
    if (dev.cur() == 1L) stop("Graphics device failed: ", target)
    draw(); dev.off()
    if (!file.exists(staged) || file.info(staged)$size == 0) stop("Missing figure: ", target)
    if (!file.copy(staged, target, overwrite=TRUE)) stop("Cannot write figure: ", target)
    unlink(staged)
  }
}
read_result <- function(file) read.csv(file.path(root, "results", "followup_v4", file))
colours <- c(bo_iso="#0072B2", bo_ard="#D55E00", bo_iso_pool="#009E73",
             bo_ard_pool="#CC79A7", cma_full="#6A3D9A", cma_diag="#E69F00")
labels <- c(bo_iso="Scalar, continuous", bo_ard="ARD, continuous",
            bo_iso_pool="Scalar, pool", bo_ard_pool="ARD, pool",
            cma_full="Full CMA", cma_diag="Diagonal CMA")
rotation <- read_result("rotation_intervals.csv")
draw_rotation <- function(methods, block, calls, title, ylim) {
  plot(NA, xlim=c(.55,3.45), ylim=ylim, xaxt="n", xlab="", ylab="", main=title, bty="l")
  abline(h=0, col="#707070", lty=2)
  axis(1, 1:3, c("Q404","Q505","Q606"))
  offsets <- seq(-.22,.22,length.out=length(methods))
  for (i in seq_along(methods)) {
    d <- rotation[rotation$block==block & rotation$method==methods[i] & rotation$calls==calls,]
    d <- d[order(d$orientation),]
    x <- 1:3 + offsets[i]
    segments(x,d$low,x,d$high,col=colours[methods[i]],lwd=1.8)
    segments(x-.035,d$low,x+.035,d$low,col=colours[methods[i]])
    segments(x-.035,d$high,x+.035,d$high,col=colours[methods[i]])
    points(x,d$estimate,pch=16,col=colours[methods[i]],cex=.9)
  }
}
fig2 <- function() {
  layout(matrix(c(1,2,3,4,4,4),nrow=2,byrow=TRUE),heights=c(1,.26))
  par(family="sans",cex=.93,cex.main=1,mar=c(2.3,2.6,2.5,.4),oma=c(0,2.2,0,0))
  draw_rotation(c("bo_iso","bo_ard","bo_iso_pool","bo_ard_pool"),"battery_bo",100,
                "A  BO: 100 calls",c(-.15,.31))
  draw_rotation(c("cma_full","cma_diag"),"cma_ball",100,
                "B  CMA: 100 calls",c(-.15,.31))
  draw_rotation(c("cma_full","cma_diag"),"cma_ball",300,
                "C  CMA: 300 calls",c(-.15,.31))
  mtext("Mean log10(rotated / identity MSE)",side=2,outer=TRUE,line=.6,cex=.9)
  par(mar=c(0,0,0,0)); plot.new()
  legend("center",legend=labels,col=colours,pch=16,ncol=3,bty="n",cex=.92)
}
paths <- read_result("path_distances.csv")
quality <- read_result("pool_quality_intervals.csv")
fig3 <- function() {
  par(mfrow=c(1,2),family="sans",cex=.95,cex.main=1.0,mar=c(6.3,4.3,2.5,1.0))
  methods <- c("bo_iso","bo_ard","bo_iso_pool","bo_ard_pool")
  plot(NA,xlim=c(1,100),ylim=c(-16,1),xlab="Counted evaluations",ylab="log10(physical candidate distance)",
       main="A  Battery path agreement",bty="l")
  abline(v=20,lty=3,col="#888888")
  for (m in methods) {
    d <- paths[paths$block=="battery_bo" & paths$method==m,]
    means <- aggregate(distance ~ initial_design_seed + evaluation, d, mean)
    s <- aggregate(distance ~ evaluation, means, median)
    lines(s$evaluation,log10(pmax(s$distance,1e-16)),col=colours[m],lwd=1.7)
  }
  legend("bottom",inset=c(0,-.52),legend=labels[methods],col=colours[methods],lty=1,
         ncol=2,bty="n",xpd=NA,cex=.78)
  par(mar=c(6.3,4.7,2.5,.6))
  plot(NA,xlim=c(.6,4.4),ylim=c(-.08,2.35),xaxt="n",xlab="",ylab="Mean log10(pool / continuous loss)",
       main="B  Quadratic quality cost",bty="l")
  axis(1,1:4,c("Identity","Q404","Q505","Q606"))
  abline(h=0,col="#707070",lty=2)
  for (i in 1:2) {
    kernel <- c("iso","ard")[i]
    d <- quality[quality$block=="quadratic_bo" & quality$kernel==kernel,]
    d <- d[order(d$orientation),]
    x <- 1:4 + c(-.09,.09)[i]
    co <- colours[paste0("bo_",kernel)]
    segments(x,d$low,x,d$high,col=co,lwd=1.8)
    points(x,d$estimate,pch=16,col=co)
  }
  text(1.35,1.94,"88x loss",pos=4,cex=.85,col=colours["bo_ard"])
  legend("bottom",inset=c(0,-.52),legend=c("Scalar kernel","ARD kernel"),
         col=colours[c("bo_iso","bo_ard")],pch=16,ncol=2,bty="n",xpd=NA,cex=.82)
}
for (n in c(2,3)) {
  draw <- if(n==2) fig2 else fig3
  save_figure(if(n==3) 4 else n, draw, 8)
}

v5 <- read.csv(file.path(root,"results","v5-aligned","contrasts.csv"))
fig4 <- function() {
  layout(matrix(c(1,2),nrow=1),widths=c(1.9,1))
  par(family="sans",cex=.94,mgp=c(1.8,.5,0))
  methods <- c("bo_iso","bo_ard","bo_iso_pool")
  names <- c("scalar","ARD","pool")
  par(mar=c(3.9,7.5,2.3,.8))
  plot(NA,xlim=c(.85,6),ylim=c(.45,6.65),log="x",yaxt="n",xaxt="n",ylab="",
       xlab="CMA / BO MSE ratio (log scale)",main="A  Paired method comparison",bty="n",cex.main=1)
  axis(1,at=c(1,1.5,2,3,5),labels=c("1","1.5","2","3","5"))
  axis(2,at=6:1,labels=c(paste("Resampling /",names),paste("Projection /",names)),
       las=1,tick=FALSE,cex.axis=.85)
  abline(v=1,lty=2,col="#777777"); abline(h=3.5,col="#DDDDDD")
  for (i in seq_along(methods)) for (h in 1:2) for (calls in c(100,300)) {
    key <- paste("CMA",c("resample","project")[h],"/",methods[i])
    d <- v5[v5$contrast==key & v5$calls==calls,]
    stopifnot(nrow(d)==1)
    y <- 7-i-3*(h-1) + if(calls==100) .14 else -.14
    co <- if(calls==100) "#0072B2" else "#D55E00"
    segments(10^d$low,y,10^d$high,y,col=co,lwd=1.8)
    points(10^d$estimate,y,pch=if(calls==100) 16 else 17,col=co,cex=.9)
  }
  mtext("Above 1: lower loss with BO",side=1,line=2.9,cex=.8)
  par(mar=c(3.9,1.3,2.3,.6))
  plot(NA,xlim=c(.45,1.15),ylim=c(.45,6.65),yaxt="n",xaxt="n",ylab="",
       xlab="Projection / resampling MSE",main="B  Handler effect",bty="n",cex.main=1)
  axis(1,at=c(.5,.75,1),labels=c("0.50","0.75","1.00"))
  abline(v=1,lty=2,col="#777777")
  for(calls in c(100,300)) {
    d <- v5[v5$contrast=="CMA projection / CMA resampling" & v5$calls==calls,]
    stopifnot(nrow(d)==1)
    y <- if(calls==100) 4.5 else 2.8
    co <- if(calls==100) "#0072B2" else "#D55E00"
    segments(10^d$low,y,10^d$high,y,col=co,lwd=2)
    points(10^d$estimate,y,pch=if(calls==100) 16 else 17,col=co)
    text(.48,y+.5,paste(calls,"calls"),adj=0,cex=.86)
  }
  mtext("Below 1: lower loss with projection",side=1,line=2.9,cex=.8)
}
save_figure(5, fig4, 9)
