library(ggplot2)
#heatmap of signaling type.
data <- read.csv("../data/jaaks_druginfo.csv", header = TRUE)
f_y = factor(data$drug_name, levels = c(data$drug_name))
p1 <- ggplot(data = data, aes(x = c(1), y = f_y)) + 
  geom_raster(aes(fill = factor(target_pathway)))+
  scale_fill_viridis_d(option = 'F', direction = -1, begin = 0.15)+
  theme_bw()+
  theme(text = element_text(family = "serif",size = 8), 
        axis.ticks.x = element_blank(), 
        axis.ticks.y = element_blank(), 
        axis.text.x = element_blank(), 
        axis.title.x = element_blank(),
        axis.title.y =  element_blank(),
        legend.direction = "vertical",
        legend.title = element_blank(),
        panel.border = element_blank(),
        panel.grid = element_blank())+
  guides(fill = guide_legend(reverse = TRUE))
p1
#heatmap of targets type.
data2 <- read.csv("../data/drugs_info4heatmap.csv", header = TRUE)
f_y = factor(data2$drug_name, levels = c(data$drug_name))
f_x = factor(data2$target_type)
p2 <- ggplot(data = data2, aes(x = f_x, y = f_y)) + 
  geom_raster(aes(fill = marker))+
  scale_fill_viridis_c(option = 'D', direction = -1, begin = 0.15)+
  theme_bw()+
  theme(text = element_text(family = "serif",size = 7, angle = 45),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y =  element_blank(),
        legend.direction = "vertical",
        legend.title = element_blank())+
  guides(fill = guide_legend(reverse = TRUE))
p2

#use pheatmap package.
library(pheatmap)
data1 <- read.csv("../data/jaaks_druginfo.csv", header = TRUE)
data2 <- read.csv("../data/target_drug4heatmap.csv", header = T, row.names = 1)
annotation_col <- data.frame(signal = c(data1$target_pathway))
row.names(annotation_col) <- colnames(data2)

p <- pheatmap(data2, cluster_rows = F, cluster_cols = F, 
              color = colorRampPalette(c("white","firebrick3"))(7),
              show_colnames = T, show_rownames = T,
              annotation_col = annotation_col)
p

#bar plot
target_classfied = read.csv("../data/target_classified.csv", header = T)
f_x = factor(target_classfied$drug_name, levels = unique(target_classfied$drug_name))
p <- ggplot(target_classfied, aes(f_x, fill = factor(target_type))) +
  geom_bar() +
  scale_y_continuous(expand = c(0,0))+
  theme_bw() +
  theme(text = element_text(family = "serif",size = 7),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y =  element_blank(),
        legend.direction = "vertical",
        legend.title = element_blank(), 
        panel.grid = element_blank(), 
        panel.border = element_blank())+
  guides(fill = "none")
  #guides(fill = guide_legend(ncol = 2, reverse = TRUE))
p
ggsave("target_class_nolegend.pdf", width = 12, height = 2, units = "in")
###heat map of cancer type:
library(pheatmap)
library(RColorBrewer)
data1 <- read.csv("../data/jaaks_druginfo.csv", header = T)
data2 <- read.csv("../data/cancer_info_matrix.csv", header = T, row.names = 1)
annotation_col <- data.frame(signal = c(data1$target_pathway))
row.names(annotation_col) <- colnames(data2)

p <- pheatmap(data2, cluster_rows = F, cluster_cols = F, 
              color = colorRampPalette(c("white","firebrick3"))(7),
              show_colnames = T, show_rownames = T,
              annotation_col = annotation_col)#, 
              
#filename = "cancer_info_matrix.pdf", width = 12, height = 6)
p

p <- pheatmap(data2, cluster_rows = F, cluster_cols = F, 
             color = colorRampPalette(c("white", "green", 
                                        "mediumblue", "yellow",
                                        "purple", "orchid1", 
                                        "#ff3038"))(7),
             show_colnames = T, show_rownames = T,
             annotation_col = annotation_col,fontsize = 8,
             filename = "cancer_info_matrix.pdf", width = 14, height = 5) 

#heatmap of signal pathway similarity between compounds using signaturizers.
cosine <- read.csv("../data/cosine_similarity_64cpds_sigpath.csv", header = TRUE, row.names = 1)
#mink <- read.csv("../data/minkowski_similarity_64cpds_mechanism.csv", header = T, row.names = 1)
pearson <- read.csv("../data/pearsonr_similarity_64cpds_sigpath.csv", header = T, row.names = 1)
data <- read.csv("../data/jaaks_druginfo.csv", header = TRUE)
annotation_col <- data.frame(signal = c(data$target_pathway))
annotation_row <- data.frame(signal = c(data$target_pathway))
row.names(annotation_col) <- colnames(pearson)
row.names(annotation_row) <- rownames(pearson)
p <- pheatmap(cosine, cluster_rows = F, cluster_cols = F, 
              color = colorRampPalette(c("white","firebrick"))(100),
              show_colnames = T, show_rownames = T, 
              annotation_row = annotation_row, annotation_col = annotation_col, 
              legend_breaks = c(-0.2, 0, 0.5, 1),
              filename = "../figures/pheatmap_sigpath_simi_cosine_small_copy.pdf", 
              width = 12, height = 7, annotation_legend = T)
pearson[pearson < 0.3] <- 0









