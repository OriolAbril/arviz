(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("6fc8875d-0f3d-4da2-8f43-b2e40a4ec839");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '6fc8875d-0f3d-4da2-8f43-b2e40a4ec839' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"4374c478-132e-44e3-9130-626144022f72":{"roots":{"references":[{"attributes":{"callback":null},"id":"21880","type":"HoverTool"},{"attributes":{},"id":"21866","type":"BasicTicker"},{"attributes":{},"id":"21959","type":"BasicTickFormatter"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21881","type":"BoxAnnotation"},{"attributes":{},"id":"21975","type":"UnionRenderers"},{"attributes":{},"id":"21976","type":"Selection"},{"attributes":{},"id":"21863","type":"LinearScale"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"21957"},"ticker":{"id":"21870"}},"id":"21869","type":"LinearAxis"},{"attributes":{"data":{"rug_x":{"__ndarray__":"jQwCEA1Gsz9bBMLb9PjIP1Qd9Zram7E/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/aF85wG2piz9oXznAbamLP2hfOcBtqYs/aF85wG2piz9oXznAbamLP2hfOcBtqYs/pI3yRkqEyT9c/+ob+nG6P4OiBeyjALU/mx4fY+a33D9wF2c1cbTRPy51Aws2htg/P07TEgOYwz9FOncTGHy5P/BQ+ANPucc/uRO6PYJJzj9CRCWTDYpmP+UvVnGFsrI/8JR71fNwcj8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"21964"},"selection_policy":{"id":"21963"}},"id":"21937","type":"ColumnDataSource"},{"attributes":{},"id":"21961","type":"UnionRenderers"},{"attributes":{"source":{"id":"21937"}},"id":"21939","type":"CDSView"},{"attributes":{"source":{"id":"21928"}},"id":"21932","type":"CDSView"},{"attributes":{},"id":"21962","type":"Selection"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21882","type":"PolyAnnotation"},{"attributes":{"axis":{"id":"21869"},"dimension":1,"ticker":null},"id":"21872","type":"Grid"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21936","type":"Dash"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2515582690238702},"id":"21933","type":"Span"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.1782444431478369},"id":"21934","type":"Span"},{"attributes":{"data_source":{"id":"21937"},"glyph":{"id":"21936"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21939"}},"id":"21938","type":"GlyphRenderer"},{"attributes":{"text":"tau"},"id":"21940","type":"Title"},{"attributes":{"below":[{"id":"21901"}],"center":[{"id":"21904"},{"id":"21908"}],"left":[{"id":"21905"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21945"},{"id":"21947"},{"id":"21948"},{"id":"21949"},{"id":"21952"}],"title":{"id":"21954"},"toolbar":{"id":"21919"},"toolbar_location":null,"x_range":{"id":"21893"},"x_scale":{"id":"21897"},"y_range":{"id":"21895"},"y_scale":{"id":"21899"}},"id":"21892","subtype":"Figure","type":"Plot"},{"attributes":{"toolbars":[{"id":"21883"},{"id":"21919"}],"tools":[{"id":"21873"},{"id":"21874"},{"id":"21875"},{"id":"21876"},{"id":"21877"},{"id":"21878"},{"id":"21879"},{"id":"21880"},{"id":"21909"},{"id":"21910"},{"id":"21911"},{"id":"21912"},{"id":"21913"},{"id":"21914"},{"id":"21915"},{"id":"21916"}]},"id":"21981","type":"ProxyToolbar"},{"attributes":{},"id":"21963","type":"UnionRenderers"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21935","type":"Span"},{"attributes":{"data":{"rug_x":{"__ndarray__":"fV36E1z/6j89DycBWWfXP73VlTJ7YsE/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/Oqay45Jr6D86SHRMZcflP7HThSU1z+I/VdSaCTtd6D9Hvab2ZmSwP8UA5kQ6d4M/0GULqag1oz+aI/Yi4T7rP55DEK8H/NA/qgGd6qjX1D+VqnS/h2ThP/UnuP7VN+Q/TGXHJdeQ2z8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"21976"},"selection_policy":{"id":"21975"}},"id":"21951","type":"ColumnDataSource"},{"attributes":{},"id":"21964","type":"Selection"},{"attributes":{"below":[{"id":"21865"}],"center":[{"id":"21868"},{"id":"21872"}],"left":[{"id":"21869"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21931"},{"id":"21933"},{"id":"21934"},{"id":"21935"},{"id":"21938"}],"title":{"id":"21940"},"toolbar":{"id":"21883"},"toolbar_location":null,"x_range":{"id":"21857"},"x_scale":{"id":"21861"},"y_range":{"id":"21859"},"y_scale":{"id":"21863"}},"id":"21856","subtype":"Figure","type":"Plot"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21944","type":"Circle"},{"attributes":{"children":[{"id":"21982"},{"id":"21980"}]},"id":"21983","type":"Column"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21943","type":"Circle"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"bcJe+Pxh3D88Eu5SUq3NPyL+eQAns8s/mzeOVbYJ1z+YvNPsmzTWP87j+UyletE/sIr1s8Bnzz8YN/msnr7PP8gPs0h4ec8/RNGUEkZ90z8g12riYrDUP+AlTxYjYc4/UBTOvdhAzD+AFwpyJ0DOPxDSzcUXbc4/kDNyJsikyj9AY0p3Si3PP8AWdsF70MQ/4I8dAXxLyD+grhW5nZrSPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"21974"},"selection_policy":{"id":"21973"}},"id":"21942","type":"ColumnDataSource"},{"attributes":{},"id":"21861","type":"LinearScale"},{"attributes":{"source":{"id":"21951"}},"id":"21953","type":"CDSView"},{"attributes":{"data_source":{"id":"21942"},"glyph":{"id":"21943"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21944"},"selection_glyph":null,"view":{"id":"21946"}},"id":"21945","type":"GlyphRenderer"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"TOY7mm62yD/upazjteTNP/DqXfy8Lcs/HHCIRHVOzT9orzckCyTOP6y8CkfbP8w/iNx/9NF5yz+IfM+LFJ/NPygi8KloJ8w/oK0q3zNVzD/YcLH58jHPP6hWLqhoMNQ/IGaMl5nu0j8AebTCVJLRP+D0hUiIM80/uNpKBpu90z/A00Fq0J3TP2iCFjW8ldY/kLBgWUSo1j8QCPgZgPnXPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"21962"},"selection_policy":{"id":"21961"}},"id":"21928","type":"ColumnDataSource"},{"attributes":{"source":{"id":"21942"}},"id":"21946","type":"CDSView"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21950","type":"Dash"},{"attributes":{"data_source":{"id":"21928"},"glyph":{"id":"21929"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21930"},"selection_glyph":null,"view":{"id":"21932"}},"id":"21931","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2148430013731262},"id":"21947","type":"Span"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"21971"},"ticker":{"id":"21902"}},"id":"21901","type":"LinearAxis"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.15209716424958658},"id":"21948","type":"Span"},{"attributes":{"end":1,"start":-0.05},"id":"21895","type":"DataRange1d"},{"attributes":{"data_source":{"id":"21951"},"glyph":{"id":"21950"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21953"}},"id":"21952","type":"GlyphRenderer"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"21969"},"ticker":{"id":"21906"}},"id":"21905","type":"LinearAxis"},{"attributes":{"text":"mu"},"id":"21954","type":"Title"},{"attributes":{},"id":"21897","type":"LinearScale"},{"attributes":{},"id":"21899","type":"LinearScale"},{"attributes":{"callback":null},"id":"21916","type":"HoverTool"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21949","type":"Span"},{"attributes":{},"id":"21902","type":"BasicTicker"},{"attributes":{"axis":{"id":"21901"},"ticker":null},"id":"21904","type":"Grid"},{"attributes":{"toolbar":{"id":"21981"},"toolbar_location":"above"},"id":"21982","type":"ToolbarBox"},{"attributes":{"axis":{"id":"21905"},"dimension":1,"ticker":null},"id":"21908","type":"Grid"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21930","type":"Circle"},{"attributes":{},"id":"21906","type":"BasicTicker"},{"attributes":{},"id":"21969","type":"BasicTickFormatter"},{"attributes":{"overlay":{"id":"21917"}},"id":"21911","type":"BoxZoomTool"},{"attributes":{},"id":"21910","type":"PanTool"},{"attributes":{},"id":"21909","type":"ResetTool"},{"attributes":{},"id":"21957","type":"BasicTickFormatter"},{"attributes":{},"id":"21915","type":"SaveTool"},{"attributes":{},"id":"21912","type":"WheelZoomTool"},{"attributes":{"axis":{"id":"21865"},"ticker":null},"id":"21868","type":"Grid"},{"attributes":{"overlay":{"id":"21918"}},"id":"21913","type":"LassoSelectTool"},{"attributes":{},"id":"21914","type":"UndoTool"},{"attributes":{"end":1,"start":-0.05},"id":"21859","type":"DataRange1d"},{"attributes":{},"id":"21870","type":"BasicTicker"},{"attributes":{},"id":"21878","type":"UndoTool"},{"attributes":{},"id":"21857","type":"DataRange1d"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21873"},{"id":"21874"},{"id":"21875"},{"id":"21876"},{"id":"21877"},{"id":"21878"},{"id":"21879"},{"id":"21880"}]},"id":"21883","type":"Toolbar"},{"attributes":{"overlay":{"id":"21881"}},"id":"21875","type":"BoxZoomTool"},{"attributes":{},"id":"21874","type":"PanTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21917","type":"BoxAnnotation"},{"attributes":{},"id":"21873","type":"ResetTool"},{"attributes":{},"id":"21879","type":"SaveTool"},{"attributes":{},"id":"21876","type":"WheelZoomTool"},{"attributes":{"overlay":{"id":"21882"}},"id":"21877","type":"LassoSelectTool"},{"attributes":{},"id":"21971","type":"BasicTickFormatter"},{"attributes":{},"id":"21893","type":"DataRange1d"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"21959"},"ticker":{"id":"21866"}},"id":"21865","type":"LinearAxis"},{"attributes":{"children":[[{"id":"21856"},0,0],[{"id":"21892"},0,1]]},"id":"21980","type":"GridBox"},{"attributes":{},"id":"21974","type":"Selection"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21918","type":"PolyAnnotation"},{"attributes":{},"id":"21973","type":"UnionRenderers"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21909"},{"id":"21910"},{"id":"21911"},{"id":"21912"},{"id":"21913"},{"id":"21914"},{"id":"21915"},{"id":"21916"}]},"id":"21919","type":"Toolbar"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21929","type":"Circle"}],"root_ids":["21983"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"4374c478-132e-44e3-9130-626144022f72","root_ids":["21983"],"roots":{"21983":"6fc8875d-0f3d-4da2-8f43-b2e40a4ec839"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();