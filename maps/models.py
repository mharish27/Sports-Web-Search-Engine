from django.db import models


class Destination(models.Model):
    destination_text = models.CharField(max_length=200)
