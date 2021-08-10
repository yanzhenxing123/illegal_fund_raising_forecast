# Generated by Django 3.2.6 on 2021-08-10 09:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('forecast', '0002_alter_testdataset_table'),
    ]

    operations = [
        migrations.AddField(
            model_name='testdataset',
            name='annual_report_info',
            field=models.FileField(blank=True, null=True, upload_to='test/%Y%m%d/'),
        ),
        migrations.AddField(
            model_name='testdataset',
            name='change_info',
            field=models.FileField(blank=True, null=True, upload_to='test/%Y%m%d/'),
        ),
        migrations.AddField(
            model_name='testdataset',
            name='news_info',
            field=models.FileField(blank=True, null=True, upload_to='test/%Y%m%d/'),
        ),
        migrations.AddField(
            model_name='testdataset',
            name='other_info',
            field=models.FileField(blank=True, null=True, upload_to='test/%Y%m%d/'),
        ),
        migrations.AddField(
            model_name='testdataset',
            name='tax_info',
            field=models.FileField(blank=True, null=True, upload_to='test/%Y%m%d/'),
        ),
        migrations.AlterField(
            model_name='testdataset',
            name='base_info',
            field=models.FileField(blank=True, null=True, upload_to='test/%Y%m%d/'),
        ),
    ]