##### Cheese
library(ggplot2)
data <- read.csv('https://raw.githubusercontent.com/jgscott/SDS383D/master/data/cheese.csv')

data_p <- reshape2::melt(data, id=c('vol', 'price', 'disp'))
ggplot(data_p, aes(x=price, y=vol, group=value, colour=as.factor(disp))) +
 geom_point()

ggplot(data_p, aes(x = value, y = vol, colour = as.factor(disp))) +
    geom_point()
