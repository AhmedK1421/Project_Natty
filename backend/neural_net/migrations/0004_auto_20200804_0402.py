# Generated by Django 3.0.8 on 2020-08-04 04:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('neural_net', '0003_user_name'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='name',
            field=models.CharField(max_length=100),
        ),
    ]
