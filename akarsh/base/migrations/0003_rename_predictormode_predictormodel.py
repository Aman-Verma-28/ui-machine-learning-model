# Generated by Django 4.2.1 on 2023-05-13 14:31

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("base", "0002_alter_predictormode_ans"),
    ]

    operations = [
        migrations.RenameModel(
            old_name="PredictorMode",
            new_name="PredictorModel",
        ),
    ]
