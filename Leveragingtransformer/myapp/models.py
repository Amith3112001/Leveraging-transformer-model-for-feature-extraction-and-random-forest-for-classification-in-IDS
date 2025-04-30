from django.db import models

# Create your models here.


class userdetails(models.Model):
    uid = models.AutoField(primary_key=True)
    uname = models.CharField(max_length=30, null=False)
    address = models.CharField(max_length=30)
    mobno = models.CharField(max_length=30, null=False)
    email = models.CharField(max_length=30, null=False)
    pwd = models.CharField(max_length=30, null=False)

    def __str__(self):
        return "%s %s %s %s %s %s" % (
            self.uid, self.uname, self.address, self.mobno, self.email, self.pwd)
