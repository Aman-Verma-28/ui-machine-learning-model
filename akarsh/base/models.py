from django.db import models

from base.utils import predictor

# Create your models here.
class PredictorModel(models.Model):
    ia = models.FloatField()
    ib = models.FloatField()
    ic = models.FloatField()
    va = models.FloatField()
    vb = models.FloatField()
    vc = models.FloatField()
    fault_type = models.CharField(null=True, blank=True, max_length=50)

    def save(self, *args, **kwargs):
        ans = predictor(self.ia, self.ib, self.ic, self.va, self.vb, self.vc)
        self.fault_type = ans
        super().save()

    def __str__(self) -> str:
        return str(self.id)
    
class GuideLines(models.Model):
    number = models.IntegerField()
    guide_line = models.TextField()

    def __str__(self) -> str:
        return "Guide Line {}".format(self.number)




        

    
