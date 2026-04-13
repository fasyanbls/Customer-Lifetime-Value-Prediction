from django.core.management.base import BaseCommand, CommandError
from analytics.ml_model import train_clv_model, compare_models, get_feature_importance
import time


class Command(BaseCommand):
    help = 'Train the CLV Gradient Boosting model, or compare Linear Regression vs Random Forest vs Gradient Boosting'

    def add_arguments(self, parser):
        parser.add_argument(
            '--compare',
            action='store_true',
            help='Compare all 3 models before training the final one',
        )
        parser.add_argument(
            '--show-importance',
            action='store_true',
            help='Print feature importance scores after training',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.HTTP_INFO(
            '\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'
        ))
        self.stdout.write(self.style.HTTP_INFO(
            '   CLV Analytics — ML Model Training'
        ))
        self.stdout.write(self.style.HTTP_INFO(
            '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
        ))

        # ── Optional: compare models first ──
        if options['compare']:
            self.stdout.write('  Comparing 3 models on 80/20 chronological split + 5-fold CV...\n')
            cmp = compare_models()
            if not cmp.get('success'):
                raise CommandError(f'Comparison failed: {cmp.get("error")}')

            self.stdout.write(self.style.HTTP_INFO(
                f'  {"Model":<25} {"MAE ($)":<12} {"R² Test":<12} {"CV R²":<16} {"Verdict"}'
            ))
            self.stdout.write('  ' + '─' * 75)

            for name, scores in cmp['results'].items():
                verdict = (
                    '★ Best — use this'    if name == 'Gradient Boosting' else
                    'Good but slower'      if name == 'Random Forest'     else
                    'Baseline (simplest)'
                )
                self.stdout.write(
                    f'  {name:<25} ${scores["mae"]:<11} {scores["r2"]:<12} '
                    f'{scores["cv_r2_mean"]}±{scores["cv_r2_std"]:<10} {verdict}'
                )
            self.stdout.write('')

        # ── Train final model ──
        self.stdout.write('  Training Gradient Boosting on full training set (80%)...')
        self.stdout.write('  Evaluating on held-out test set (20%)...\n')

        start = time.time()
        result = train_clv_model()
        elapsed = round(time.time() - start, 2)

        if result.get('success'):
            self.stdout.write(self.style.SUCCESS(
                f'  ✓ Training completed in {elapsed}s'
            ))
            self.stdout.write(self.style.SUCCESS(
                f'    Train samples      : {result["n_train"]}'
            ))
            self.stdout.write(self.style.SUCCESS(
                f'    Test  samples      : {result["n_test"]}'
            ))
            self.stdout.write(self.style.SUCCESS(
                f'    MAE  ($ error)     : ${result["mae"]}'
            ))
            self.stdout.write(self.style.SUCCESS(
                f'    RMSE               : ${result["rmse"]}'
            ))
            self.stdout.write(self.style.SUCCESS(
                f'    R² on test set     : {result["r2"]}'
            ))
            self.stdout.write(self.style.SUCCESS(
                f'    Cross-Val R²       : {result["cv_r2_mean"]} ± {result["cv_r2_std"]}'
            ))
            self.stdout.write(self.style.SUCCESS(
                f'    Model saved to     : models/clv_model.pkl'
            ))
            self.stdout.write(self.style.SUCCESS(
                f'    DB Log ID          : #{result["model_id"]}'
            ))

            # Interpret R²
            r2 = result['r2']
            if   r2 >= 0.85: quality = 'Excellent fit'
            elif r2 >= 0.70: quality = 'Good fit'
            elif r2 >= 0.50: quality = 'Moderate — consider more features'
            else:             quality = 'Weak — check data quality'
            self.stdout.write(self.style.HTTP_INFO(f'\n  Model quality: {quality} (R²={r2})'))

            if options['show_importance']:
                fi = get_feature_importance()
                if fi:
                    self.stdout.write('\n  Feature Importance:')
                    for feat, score in fi.items():
                        bar = '█' * int(score * 40)
                        self.stdout.write(f'    {feat:<28} {bar} {score*100:.1f}%')

            self.stdout.write(self.style.HTTP_INFO(
                '\n  Next step: python manage.py predict_customers\n'
            ))
        else:
            raise CommandError(f'Training failed: {result.get("error")}')
