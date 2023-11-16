library(ggplot2)
target_classfied = read.csv("data/one_graph/bardata_1362_targets_1stlayer_v2.csv", header = T)
f_x = factor(target_classfied$compounds, levels = unique(target_classfied$compounds))
p <- ggplot(target_classfied, aes(f_x, fill = factor(targets))) +
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
        panel.border = element_blank())#+
  #guides(fill = "none")
p
ggsave("figures/barplot_1362_targets_1stlayer_v2.pdf", dpi = 600)
