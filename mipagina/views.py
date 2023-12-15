from django.shortcuts import render
from django.core.mail import send_mail
from django.conf import settings

# Create your views here.

def inicio(request):
    if  request.method == 'POST':
        nombre=request.POST.get('nombre')
        mensaje=request.POST.get('mensaje')
        email=request.POST.get('email')
        print(nombre)
        send_mail(
            'Contacto',
            f'''{mensaje} mensaje enviado por: {nombre} correo: {email}''',
            'settings.EMAIL_HOST_USER',
            [email],
            fail_silently=False
        )
        send_mail(
            'Contacto',
            f'''{mensaje} mensaje enviado por: {nombre} correo: {email}''',
            'settings.EMAIL_HOST_USER',
            ['oscar.r.server@gmail.com'],
            fail_silently=False
        )
        return render(request,'inicio.html')
    return render(request,'inicio.html')

