df<-read.csv("Car_sales.csv")
df
car <- na.omit(df)
car

#run model with all features 
mlr<-lm(Fuel_efficiency ~ Sales_in_thousands + Price_in_thousands + Engine_size + Horsepower + Wheelbase + Width + Length + Curb_weight + Fuel_capacity, data = car)
summary(mlr)
anova(mlr)

install.packages("car")
library(car)

#ran model without horsepower and performance factor due to high correlation between HorsePower and Engine; Performance Factor and Price
mlr<-lm(Fuel_efficiency ~ Price_in_thousands + Engine_size  + Width + Length +Wheelbase+ Curb_weight + Fuel_capacity, data = car)
summary(mlr)
anova(mlr)

step(null, scope=list(lower=null, upper=full), direction="forward", alpha = 0.05)
step(full, scope=list(lower=null, upper=full), direction="backward", alpha = 0.05)
step(null, scope = list(upper=full), data=heat, direction="both")

#With all 3 model are the same, we are using the backward model for linear regression model building
backwardmodel = lm(Fuel_efficiency ~ Engine_size +  Curb_weight + Fuel_capacity  + Length, data = car)
summary(backwardmodel)
anova(backwardmodel)

#normality plot of residuals step
res <- resid(backwardmodel)
qqnorm(res)
qqline(res)

#plot residuals
plot(fitted(backwardmodel),res)
plot(fitted(backwardmodel),car$Fuel_efficiency)
abline(0,0)

Residual=residuals(backwardmodel) 			## RESIDUAL
Stand_Res=stdres(backwardmodel)			## STANDARDIZED RESIDUAL
Student_Res=studres(backwardmodel)   		## STUDENTIZED RESIDUAL 
R_Student = rstudent(backwardmodel) 		## COMPUTING R-Student
Lev_hii=hatvalues(backwardmodel)             	## LEVERAGE POINTS 
CookD=cooks.distance(backwardmodel)  		## COOKS DISTANCE 

allres2=cbind.data.frame(car$Fuel_efficiency,Residual,Stand_Res,Student_Res,R_Student,Lev_hii,CookD)

#Detect Outlier by using R-student and Cooks D
allres2$I_RStu=ifelse(abs(allres2$R_Student) > qt(0.975,147), c("Inspect"), c("0"))
allres2$I_CookD =  ifelse(allres2$CookD > qf(0.5,6,147), c("Inspect"), c("0"))
allres2

#remove outliers
t = rstudent(backwardmodel) 
t
car2=car[abs(t) <= qt(0.975,147),]
car2

#rerun model without outliers
backwardmodel2 = lm(Fuel_efficiency ~ Engine_size +  Curb_weight + Fuel_capacity  + Length, data = car2)
summary(backwardmodel2)
anova(backwardmodel2)

#normality plot of residuals step
res <- resid(backwardmodel2)
qqnorm(res)
qqline(res)

#plot residuals
plot(fitted(backwardmodel2),res)
abline(0,0)

#plot y v. yhat values
plot(car2$Fuel_efficiency,exp(fitted(backwardmodel2)), xlab = "Y", ylab = "Yhat")
abline(0,1)

#transform target variable to log(y) and rerun summary and residual plots
#by using log transformation the number of data point of target value y increase from 5 to 11 which normalize the model
backwardmodel2 = lm(log(Fuel_efficiency) ~ Engine_size +  Curb_weight + Fuel_capacity  + Length, data = car2)
summary(backwardmodel2)
anova(backwardmodel2)

#normality plot of residuals step
res <- resid(backwardmodel2)
qqnorm(res)
qqline(res)

#plot residuals
plot(fitted(backwardmodel2),res)
abline(0,0)

#plot y v. yhat values
plot(car2$Fuel_efficiency,exp(fitted(backwardmodel2)), xlab = "Y", ylab = "Yhat")
abline(0,1)


#Kfold Cross Validation and K-fold Model to test run the model and apply it to different Random Dataset
library(DAAG)
library(ggplot2)
library(lattice)
library(caret)
library(DAAG)
library(robustbase)
library(cvTools)

set.seed(30)

folds=cvFolds(n=nrow(car2),K=5)

cv.lm(data=car2,form.lm=backwardmodel2,m=5) 

CV.model1=cvLm(backwardmodel2,folds=folds) 
CV.model1$cv

cv.lm(data=car2,form.lm=backwardmodel2,m=5)


anova(backwardmodel2)

train_control= trainControl(method="cv",number=5) 

model.1=train(log(Fuel_efficiency)~Engine_size +  Curb_weight + Fuel_capacity  + Length,data=car2,trControl =train_control,method="lm") 

model.1$resample
