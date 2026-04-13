from django.core.management.base import BaseCommand, CommandError
from analytics.ml_model import predict_all_customers
import time


class Command(BaseCommand):
    help = 'Predict CLV for all customers in the OLAP fact table and update clv_predicted + clv_segment columns'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be updated without actually writing to the database',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        self.stdout.write(self.style.HTTP_INFO(
            '\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
        )
        self.stdout.write(self.style.HTTP_INFO(
            '   CLV Analytics — Batch Prediction')
        )
        self.stdout.write(self.style.HTTP_INFO(
            '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n')
        )

        if dry_run:
            self.stdout.write(self.style.WARNING(
                '  [DRY RUN] No changes will be written to the database.\n'
            ))

        self.stdout.write('  Loading trained model from models/clv_model.pkl...')
        self.stdout.write('  Running predictions for all customers in fact table...')

        start_time = time.time()

        if dry_run:
            # Just count without updating
            from analytics.models import FactCustomerCLV
            count = FactCustomerCLV.objects.count()
            self.stdout.write(self.style.WARNING(
                f'\n  [DRY RUN] Would update {count:,} customer records.'
            ))
            self.stdout.write(self.style.WARNING(
                '  Run without --dry-run to apply changes.\n'
            ))
            return

        result = predict_all_customers()
        elapsed = round(time.time() - start_time, 2)

        if result.get('success'):
            self.stdout.write('')
            self.stdout.write(self.style.SUCCESS(
                f'  ✓ Predictions completed in {elapsed}s'
            ))
            self.stdout.write(self.style.SUCCESS(
                f'    Customers updated: {result["updated"]:,}'
            ))

            # Show segment distribution
            from analytics.models import FactCustomerCLV
            from django.db.models import Count
            segments = FactCustomerCLV.objects.values('clv_segment').annotate(
                count=Count('fact_id')
            ).order_by('-count')

            self.stdout.write('')
            self.stdout.write('  Segment distribution after prediction:')
            for seg in segments:
                label = seg['clv_segment']
                count = seg['count']
                bar = '█' * min(count // 2, 30)
                self.stdout.write(f'    {label:<10} {bar} {count}')

            self.stdout.write(self.style.HTTP_INFO(
                '\n  Done! Visit http://127.0.0.1:8000/ to see the dashboard.\n'
            ))
        else:
            raise CommandError(f'Prediction failed: {result.get("error")}')
